# Copyright 2023 OmniSafeAI Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torch.distributions import Normal
from safepo.utils.act import ACTLayer
from safepo.utils.mlp import MLPBase
from safepo.utils.util import check, init
from safepo.utils.util import get_shape_from_obs_space


def quantile_regression_loss(predicted_quantiles, target_values, tau):
    """
    Compute the quantile regression loss for IQN training.
    
    This loss function is used to train the IQN to predict the correct quantiles
    of the target distribution.
    
    Args:
        predicted_quantiles (torch.Tensor): Predicted quantile values from IQN
        target_values (torch.Tensor): Target values (ground truth)
        tau (torch.Tensor): Quantile levels used for prediction
        
    Returns:
        torch.Tensor: Quantile regression loss
    """
    # Expand target values to match predicted quantiles shape
    target_expanded = target_values.unsqueeze(-1).expand_as(predicted_quantiles)
    
    # Compute quantile regression loss
    # L = max(τ(y - ŷ), (1-τ)(ŷ - y))
    diff = target_expanded - predicted_quantiles
    loss = torch.where(diff >= 0, tau * diff, (tau - 1) * diff)
    
    return loss.mean()


def build_mlp_network(sizes, use_layer_norm=False):
    """
    Build a multi-layer perceptron (MLP) neural network.

    This function constructs an MLP network with the specified layer sizes and activation functions.

    Args:
        sizes (list of int): List of integers representing the sizes of each layer in the network.
        use_layer_norm (bool): Whether to use layer normalization after each linear layer.

    Returns:
        nn.Sequential: An instance of PyTorch's Sequential module representing the constructed MLP.
    """
    layers = list()
    for j in range(len(sizes) - 1):
        act = nn.Tanh if j < len(sizes) - 2 else nn.Identity
        affine_layer = nn.Linear(sizes[j], sizes[j + 1])
        nn.init.kaiming_uniform_(affine_layer.weight, a=np.sqrt(5))
        layers += [affine_layer]
        if use_layer_norm and j < len(sizes) - 2:
            layers += [nn.LayerNorm(sizes[j + 1])]
        layers += [act()]
    return nn.Sequential(*layers)


class IQNCritic(nn.Module):
    """
    Implicit Quantile Networks (IQN) critic for cost estimation.
    
    This class implements an IQN that can estimate the full distribution of cost values
    and compute CVaR (Conditional Value at Risk) for risk-aware cost estimation.
    
    Args:
        obs_dim (int): Dimensionality of the observation space.
        hidden_sizes (list): List of hidden layer sizes.
        num_quantiles (int): Number of quantiles to sample for distribution estimation.
        embedding_dim (int): Dimension of the quantile embedding.
        use_risk (bool): Whether to use risk information.
        risk_size (int): Size of risk information.
        use_layer_norm (bool): Whether to use layer normalization.
    """
    
    def __init__(self, obs_dim, hidden_sizes: list = [64, 64], num_quantiles=32, 
                 embedding_dim=64, use_risk=False, risk_size=None, use_layer_norm=False):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.embedding_dim = embedding_dim
        self.use_risk = use_risk
        
        # Quantile embedding network
        self.quantile_embedding = nn.Sequential(
            nn.Linear(embedding_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[0])
        )
        
        # Main network
        if use_risk:
            input_dim = obs_dim + risk_size + hidden_sizes[0]  # obs + risk + quantile_embedding
        else:
            input_dim = obs_dim + hidden_sizes[0]  # obs + quantile_embedding
            
        self.main_network = build_mlp_network([input_dim] + hidden_sizes[1:] + [1], use_layer_norm=use_layer_norm)
        
    def forward(self, obs, risk=None, tau=None):
        """
        Forward pass of the IQN critic.
        
        Args:
            obs (torch.Tensor): Observation tensor of shape (batch_size, obs_dim)
            risk (torch.Tensor, optional): Risk tensor of shape (batch_size, risk_size)
            tau (torch.Tensor, optional): Quantile levels of shape (batch_size, num_quantiles)
            
        Returns:
            torch.Tensor: Quantile values of shape (batch_size, num_quantiles)
        """
        # Ensure obs is 2D
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)  # Add batch dimension if missing
        
        batch_size = obs.shape[0]
        
        # Generate random quantiles if not provided
        if tau is None:
            tau = torch.rand(batch_size, self.num_quantiles, device=obs.device)
        
        # Ensure tau has the correct batch size
        if tau.shape[0] != batch_size:
            # If tau has a different batch size, adjust it
            if tau.shape[0] == 1:
                # If tau is a single sample, repeat it for the batch
                tau = tau.repeat(batch_size, 1)
            else:
                # Otherwise, take the first batch_size samples
                tau = tau[:batch_size]
        
        # Get quantile embedding
        tau_embedding = self._get_quantile_embedding(tau)  # (batch_size, num_quantiles, embedding_dim)
        num_quantiles = tau.shape[1]
        
        # Expand obs and risk to match quantile dimensions
        # Use repeat instead of expand to ensure proper broadcasting
        # Ensure obs has the correct shape before expansion
        if len(obs.shape) != 2:
            raise ValueError(f"Expected obs to be 2D, got shape {obs.shape}")
        
        obs_expanded = obs.unsqueeze(1).repeat(1, num_quantiles, 1)  # (batch_size, num_quantiles, obs_dim)
        
        # Concatenate inputs
        if self.use_risk and risk is not None:
            # Ensure risk is 2D
            if len(risk.shape) == 1:
                risk = risk.unsqueeze(0)  # Add batch dimension if missing
                
            # Ensure risk has the correct batch size
            if risk.shape[0] != batch_size:
                if risk.shape[0] == 1:
                    risk = risk.repeat(batch_size, 1)
                else:
                    risk = risk[:batch_size]
            risk_expanded = risk.unsqueeze(1).repeat(1, num_quantiles, 1)  # (batch_size, num_quantiles, risk_size)
            
            # Ensure all tensors have the same first two dimensions before concatenation
            assert obs_expanded.shape[:2] == risk_expanded.shape[:2] == tau_embedding.shape[:2], \
                f"Shape mismatch: obs_expanded {obs_expanded.shape}, risk_expanded {risk_expanded.shape}, tau_embedding {tau_embedding.shape}"
            
            x = torch.cat([obs_expanded, risk_expanded, tau_embedding], dim=-1)
        else:
            # Ensure all tensors have the same first two dimensions before concatenation
            assert obs_expanded.shape[:2] == tau_embedding.shape[:2], \
                f"Shape mismatch: obs_expanded {obs_expanded.shape}, tau_embedding {tau_embedding.shape}"
            
            x = torch.cat([obs_expanded, tau_embedding], dim=-1)
            
        # Forward through main network
        quantile_values = self.main_network(x)  # (batch_size, num_quantiles, 1)
        
        return quantile_values.squeeze(-1)  # Shape: (batch_size, num_quantiles)
    
    def _get_quantile_embedding(self, tau):
        """
        Create quantile embedding using cosine basis functions.
        
        Args:
            tau (torch.Tensor): Quantile levels of shape (batch_size, num_quantiles)
            
        Returns:
            torch.Tensor: Quantile embedding of shape (batch_size, num_quantiles, embedding_dim)
        """
        # Ensure tau has the expected shape
        if len(tau.shape) != 2:
            raise ValueError(f"Expected tau to have 2 dimensions, got {len(tau.shape)}")
        
        batch_size, num_quantiles = tau.shape
        
        # Expand tau to create embedding
        tau_expanded = tau.unsqueeze(-1)  # (batch_size, num_quantiles, 1)
        
        # Create cosine basis functions
        cos_basis = torch.cos(torch.arange(self.embedding_dim, device=tau.device) * np.pi * tau_expanded)
        # cos_basis shape: (batch_size, num_quantiles, embedding_dim)
        
        # Apply embedding network to each quantile separately
        cos_basis_flat = cos_basis.view(-1, self.embedding_dim)  # (batch_size * num_quantiles, embedding_dim)
        
        embedding_flat = self.quantile_embedding(cos_basis_flat)  # (batch_size * num_quantiles, hidden_size)
        embedding = embedding_flat.view(batch_size, num_quantiles, -1)  # (batch_size, num_quantiles, hidden_size)
        
        # Ensure the output has the correct shape
        expected_shape = (batch_size, num_quantiles, self.quantile_embedding[-1].out_features)
        if embedding.shape != expected_shape:
            raise ValueError(f"Expected embedding shape {expected_shape}, got {embedding.shape}")
        
        return embedding
    
    def get_cvar(self, obs, risk=None, alpha=0.1, num_samples=None, is_cost=True):
        """
        Compute CVaR (Conditional Value at Risk) for given observations.
        
        Args:
            obs (torch.Tensor): Observation tensor
            risk (torch.Tensor, optional): Risk tensor
            alpha (float): Risk level (e.g., 0.1 for 10% worst case)
            num_samples (int, optional): Number of quantile samples for estimation. 
                                       If None, uses self.num_quantiles.
            is_cost (bool): Whether this is for cost estimation (True) or reward estimation (False).
                           For costs, we look at the upper tail (higher values = worse).
                           For rewards, we look at the lower tail (lower values = worse).
            
        Returns:
            torch.Tensor: CVaR values of shape (batch_size,) or scalar if single observation
        """
        # Store original shape to determine output shape
        was_1d = len(obs.shape) == 1
        
        # Ensure obs is 2D
        if was_1d:
            obs = obs.unsqueeze(0)  # Add batch dimension if missing
            
        batch_size = obs.shape[0]
        
        # Use default num_quantiles if num_samples not specified
        if num_samples is None:
            num_samples = self.num_quantiles
        
        # Sample quantiles for estimation with correct batch size
        tau = torch.rand(batch_size, num_samples, device=obs.device)
        
        # Get quantile values
        quantile_values = self.forward(obs, risk, tau)  # (batch_size, num_samples)
        
        # Sort values to find the alpha-quantile
        sorted_values, _ = torch.sort(quantile_values, dim=-1)
        
        # Find the index corresponding to alpha
        alpha_idx = int(alpha * num_samples)
        
        if is_cost:
            # For costs: CVaR is the mean of the WORST alpha% outcomes (upper tail)
            # Take the last alpha_idx values (highest values = worst outcomes)
            cvar = torch.mean(sorted_values[:, -alpha_idx:], dim=-1)
        else:
            # For rewards: CVaR is the mean of the WORST alpha% outcomes (lower tail)
            # Take the first alpha_idx values (lowest values = worst outcomes)
            cvar = torch.mean(sorted_values[:, :alpha_idx], dim=-1)
        
        # If input was 1D, return scalar; otherwise return batch
        if was_1d:
            return cvar.squeeze(0)  # Remove batch dimension for single observation
        else:
            return cvar
    
    def get_expected_value(self, obs, risk=None, num_samples=None):
        """
        Compute expected value (mean) of the cost distribution.
        
        Args:
            obs (torch.Tensor): Observation tensor
            risk (torch.Tensor, optional): Risk tensor
            num_samples (int, optional): Number of quantile samples for estimation.
                                       If None, uses self.num_quantiles.
            
        Returns:
            torch.Tensor: Expected values of shape (batch_size,) or scalar if single observation
        """
        # Store original shape to determine output shape
        was_1d = len(obs.shape) == 1
        
        # Ensure obs is 2D
        if was_1d:
            obs = obs.unsqueeze(0)  # Add batch dimension if missing
            
        batch_size = obs.shape[0]
        
        # Use default num_quantiles if num_samples not specified
        if num_samples is None:
            num_samples = self.num_quantiles
        
        # Sample quantiles for estimation with correct batch size
        tau = torch.rand(batch_size, num_samples, device=obs.device)
        
        # Get quantile values
        quantile_values = self.forward(obs, risk, tau)  # (batch_size, num_samples)
        
        # Expected value is the mean of all quantiles
        expected_value = torch.mean(quantile_values, dim=-1)
        
        # If input was 1D, return scalar; otherwise return batch
        if was_1d:
            return expected_value.squeeze(0)  # Remove batch dimension for single observation
        else:
            return expected_value


class RiskEst(nn.Module):
    def __init__(self, obs_dim: int, risk_dim: int, hidden_sizes: list = [64, 64]):
        super().__init__()
        self.risk_model = build_mlp_network([obs_dim]+hidden_sizes+[risk_dim])
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, obs: torch.Tensor):
        x = self.risk_model(obs)
        return self.logsoftmax(x)


class RiskNet(nn.Module):
    def __init__(self, sizes, risk_size, use_layer_norm=False):
        super().__init__()
        self.affine_obs = nn.Linear(sizes[0], sizes[1])
        self.affine_risk = nn.Linear(risk_size, 12)
        self.activation = nn.Tanh()
        self.use_layer_norm = use_layer_norm

        if use_layer_norm:
            self.obs_norm = nn.LayerNorm(sizes[1])
            self.risk_norm = nn.LayerNorm(12)

        sizes[1] += 12
        self.rest = build_mlp_network(sizes[1:], use_layer_norm=use_layer_norm)

    def forward(self, x, risk):
        obs = self.activation(self.affine_obs(x))
        risk = self.activation(self.affine_risk(risk))
        
        if self.use_layer_norm:
            obs = self.obs_norm(obs)
            risk = self.risk_norm(risk)
            
        x = torch.cat([obs, risk], axis=-1)
        return self.rest(x)


def build_risk_mlp_network(sizes, risk_size, use_layer_norm=False):
    """
    Build a multi-layer perceptron (MLP) neural network with risk information.

    This function constructs an MLP network with the specified layer sizes and activation functions,
    incorporating risk information into the network.

    Args:
        sizes (list of int): List of integers representing the sizes of each layer in the network.
        risk_size (int): Size of the risk information input.
        use_layer_norm (bool): Whether to use layer normalization after each linear layer.

    Returns:
        RiskNet: An instance of the RiskNet module representing the constructed MLP with risk information.
    """
    return RiskNet(sizes, risk_size, use_layer_norm=use_layer_norm)


class Actor(nn.Module):
    """
    Actor network for policy-based reinforcement learning.

    This class represents an actor network that outputs a distribution over actions given observations.

    Args:
        obs_dim (int): Dimensionality of the observation space.
        act_dim (int): Dimensionality of the action space.
        hidden_sizes (list): List of hidden layer sizes.
        use_risk (bool): Whether to use risk information.
        risk_size (int): Size of risk information.
        use_layer_norm (bool): Whether to use layer normalization.

    Attributes:
        mean (nn.Sequential): MLP network representing the mean of the action distribution.
        log_std (nn.Parameter): Learnable parameter representing the log standard deviation of the action distribution.

    Example:
        obs_dim = 10
        act_dim = 2
        actor = Actor(obs_dim, act_dim)
        observation = torch.randn(1, obs_dim)
        action_distribution = actor(observation)
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list = [64, 64], use_risk=False, risk_size=None, use_layer_norm=False):
        super().__init__()
        self.use_risk = use_risk
        if use_risk:
            self.mean = build_risk_mlp_network([obs_dim]+hidden_sizes+[act_dim], risk_size, use_layer_norm=use_layer_norm)
        else:
            self.mean = build_mlp_network([obs_dim]+hidden_sizes+[act_dim], use_layer_norm=use_layer_norm)
        self.log_std = nn.Parameter(torch.zeros(act_dim), requires_grad=True)

    def forward(self, obs: torch.Tensor, risk=None):
        if self.use_risk:
            mean = self.mean(obs, risk)
        else:
            mean = self.mean(obs)
        std = torch.exp(self.log_std)
        return Normal(mean, std)


class VCritic(nn.Module):
    """
    Critic network for value-based reinforcement learning.

    This class represents a critic network that estimates the value function for input observations.

    Args:
        obs_dim (int): Dimensionality of the observation space.
        hidden_sizes (list): List of hidden layer sizes.
        use_risk (bool): Whether to use risk information.
        risk_size (int): Size of risk information.
        use_layer_norm (bool): Whether to use layer normalization.

    Attributes:
        critic (nn.Sequential): MLP network representing the critic function.

    Example:
        obs_dim = 10
        critic = VCritic(obs_dim)
        observation = torch.randn(1, obs_dim)
        value_estimate = critic(observation)
    """

    def __init__(self, obs_dim, hidden_sizes: list = [64, 64], use_risk=False, risk_size=None, use_layer_norm=False):
        super().__init__()
        self.use_risk = use_risk
        if self.use_risk:
            self.critic = build_risk_mlp_network([obs_dim]+hidden_sizes+[1], risk_size)
        else:
            self.critic = build_mlp_network([obs_dim]+hidden_sizes+[1], use_layer_norm=use_layer_norm)

    def forward(self, obs, risk=None):
        if self.use_risk:
            return torch.squeeze(self.critic(obs, risk), -1)
        else:
            return torch.squeeze(self.critic(obs), -1)


class ActorVCritic(nn.Module):
    """
    Actor-critic policy for reinforcement learning.

    This class represents an actor-critic policy that includes an actor network, two critic networks for reward
    and cost estimation, and provides methods for taking policy steps and estimating values.

    Args:
        obs_dim (int): Dimensionality of the observation space.
        act_dim (int): Dimensionality of the action space.
        hidden_sizes (list): List of hidden layer sizes.
        use_risk (bool): Whether to use risk information.
        risk_size (int): Size of risk information.
        use_actor_layer_norm (bool): Whether to use layer normalization in the actor network.
        use_critic_layer_norm (bool): Whether to use layer normalization in the critic networks.
        use_iqn_for_cost (bool): Whether to use IQN for cost estimation instead of regular critic.
        iqn_config (dict): Configuration for IQN (num_quantiles, embedding_dim, alpha for CVaR).

    Example:
        obs_dim = 10
        act_dim = 2
        actor_critic = ActorVCritic(obs_dim, act_dim)
        observation = torch.randn(1, obs_dim)
        action, log_prob, reward_value, cost_value = actor_critic.step(observation)
        value_estimate = actor_critic.get_value(observation)
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes: list = [64, 64], use_risk=False, risk_size=None, 
                 use_actor_layer_norm=False, use_critic_layer_norm=False, use_iqn_for_cost=False, 
                 iqn_config=None):
        super().__init__()
        self.use_risk = use_risk
        self.use_iqn_for_cost = use_iqn_for_cost
        
        # Reward critic (always VCritic)
        self.reward_critic = VCritic(obs_dim, hidden_sizes, use_risk=use_risk, risk_size=risk_size, 
                                    use_layer_norm=use_critic_layer_norm)
        
        # Cost critic (IQN or VCritic)
        if use_iqn_for_cost:
            if iqn_config is None:
                iqn_config = {
                    'num_quantiles': 32,
                    'embedding_dim': 64,
                    'alpha': 0.1  # CVaR alpha level
                }
            self.cost_critic = IQNCritic(obs_dim, hidden_sizes, 
                                       num_quantiles=iqn_config['num_quantiles'],
                                       embedding_dim=iqn_config['embedding_dim'],
                                       use_risk=use_risk, risk_size=risk_size, 
                                       use_layer_norm=use_critic_layer_norm)
            self.iqn_alpha = iqn_config['alpha']
        else:
            self.cost_critic = VCritic(obs_dim, hidden_sizes, use_risk=use_risk, risk_size=risk_size, 
                                      use_layer_norm=use_critic_layer_norm)
        
        self.actor = Actor(obs_dim, act_dim, hidden_sizes, use_risk=use_risk, risk_size=risk_size, 
                          use_layer_norm=use_actor_layer_norm)

    def get_value(self, obs, risk=None):
        """
        Estimate the value of observations using the critic network.

        Args:
            obs (torch.Tensor): Input observation tensor.

        Returns:
            torch.Tensor: Estimated value for the input observation.
        """
        if self.use_risk:
            return self.critic(obs, risk)
        else:
            return self.critic(obs)

    def step(self, obs, risk=None, deterministic=False):
        """
        Perform a step using the policy to get action and value estimates.

        Args:
            obs (torch.Tensor): Observation tensor
            risk (torch.Tensor, optional): Risk tensor
            deterministic (bool): Whether to use deterministic actions

        Returns:
            tuple: Tuple containing action tensor, log probabilities of the action, reward value estimate,
                   and cost value estimate.
        """

        if self.use_risk:
            dist = self.actor(obs, risk)
        else:
            dist = self.actor(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        if self.use_risk:
            value_r = self.reward_critic(obs, risk)
            if self.use_iqn_for_cost:
                # Use CVaR for cost estimation
                value_c = self.cost_critic.get_cvar(obs, risk, alpha=self.iqn_alpha, is_cost=True)
            else:
                value_c = self.cost_critic(obs, risk)
        else:
            value_r = self.reward_critic(obs)
            if self.use_iqn_for_cost:
                # Use CVaR for cost estimation
                value_c = self.cost_critic.get_cvar(obs, alpha=self.iqn_alpha, is_cost=True)
            else:
                value_c = self.cost_critic(obs)
        
        # If input was a single observation (1D), ensure output is also 1D
        if len(obs.shape) == 1:
            # Remove batch dimension if it was added
            if len(value_r.shape) > 1 and value_r.shape[0] == 1:
                value_r = value_r.squeeze(0)
            if len(value_c.shape) > 1 and value_c.shape[0] == 1:
                value_c = value_c.squeeze(0)
        
        return action, log_prob, value_r, value_c

class MultiAgentActor(nn.Module):
    """
    Multi-agent actor network for reinforcement learning.

    This class represents a multi-agent actor network that takes observations as input and produces actions and
    action probabilities as outputs. It includes options for using recurrent layers and policy active masks.

    Args:
        config (dict): Configuration parameters for the actor network.
        obs_space: Observation space of the environment.
        action_space: Action space of the environment.
        device (torch.device): Device to run the network on (default is "cpu").

    Attributes:
        hidden_size (int): Size of the hidden layers.
        config (dict): Configuration parameters for the actor network.
        _gain (float): Gain factor for action scaling.
        _use_orthogonal (bool): Flag indicating whether to use orthogonal initialization.
        _use_policy_active_masks (bool): Flag indicating whether to use policy active masks.
        _use_naive_recurrent_policy (bool): Flag indicating whether to use naive recurrent policy.
        _use_recurrent_policy (bool): Flag indicating whether to use recurrent policy.
        _recurrent_N (int): Number of recurrent layers.
        tpdv (dict): Dictionary with data type and device for tensor conversion.
        
    Example:
        config = {"hidden_size": 256, "gain": 0.1, ...}
        obs_space = gym.spaces.Box(low=0, high=1, shape=(4,))
        action_space = gym.spaces.Discrete(2)
        actor = MultiAgentActor(config, obs_space, action_space)
        observation = torch.randn(1, 4)
        rnn_states = torch.zeros(1, 256)
        masks = torch.ones(1, 1)
        actions, action_log_probs, new_rnn_states = actor(observation, rnn_states, masks)
        action = torch.tensor([0])
        action_log_probs, dist_entropy = actor.evaluate_actions(observation, rnn_states, action, masks)
    """

    def __init__(self, config, obs_space, action_space, device=torch.device("cpu")):
        super(MultiAgentActor, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.config=config
        self._gain = config["gain"]
        self._use_orthogonal = config["use_orthogonal"]
        self._use_policy_active_masks = config["use_policy_active_masks"]
        self._use_naive_recurrent_policy = config["use_naive_recurrent_policy"]
        self._use_recurrent_policy = config["use_recurrent_policy"]
        self._recurrent_N = config["recurrent_N"]
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        base =  MLPBase
        self.base = base(self.config, obs_shape)
        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, self.config)

        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Perform a forward pass through the network to generate actions and log probabilities.

        Args:
            obs (torch.Tensor): Input observation tensor.
            rnn_states (torch.Tensor): Recurrent states tensor.
            masks (torch.Tensor): Mask tensor.
            available_actions (torch.Tensor, optional): Available actions tensor (default: None).
            deterministic (bool, optional): Flag indicating whether to take deterministic actions (default: False).

        Returns:
            tuple: Tuple containing action tensor, log probability tensor, and new recurrent states tensor.
        """

        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)
        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Evaluate the actions based on the network's policy.

        Args:
            obs (torch.Tensor): Input observation tensor.
            rnn_states (torch.Tensor): Recurrent states tensor.
            action (torch.Tensor): Action tensor.
            masks (torch.Tensor): Mask tensor.
            available_actions (torch.Tensor, optional): Available actions tensor (default: None).
            active_masks (torch.Tensor, optional): Active masks tensor (default: None).

        Returns:
            tuple: Tuple containing action log probabilities tensor, distribution entropy tensor,
                   action mean tensor, action standard deviation tensor, and other optional tensors.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self.config["algorithm_name"]== "macpo":
            action_log_probs, dist_entropy, action_mu, action_std, _ = self.act.evaluate_actions_trpo(actor_features,
                                                                                                   action,
                                                                                                   available_actions,
                                                                                                   active_masks=
                                                                                                   active_masks if self._use_policy_active_masks
                                                                                                   else None)
            return action_log_probs, dist_entropy, action_mu, action_std

        else:
            action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                    action, available_actions,
                                                                    active_masks=
                                                                    active_masks if self._use_policy_active_masks
                                                                    else None)

            return action_log_probs, dist_entropy


class MultiAgentCritic(nn.Module):
    """
    Multi-agent critic network.

    This class represents a multi-agent critic network used in reinforcement learning algorithms.
    It consists of a base network (CNN or MLP), recurrent layers (if applicable), and a value output layer.

    Args:
        config (dict): Configuration dictionary.
        cent_obs_space (gym.spaces.Space): Centralized observation space.
        device (torch.device): Device to use for computations (default: cuda:0).

    Attributes:
        hidden_size (int): Size of the hidden layer.
        _use_orthogonal (bool): Flag indicating whether to use orthogonal initialization.
        _use_naive_recurrent_policy (bool): Flag indicating whether to use naive recurrent policy.
        _use_recurrent_policy (bool): Flag indicating whether to use recurrent policy.
        _recurrent_N (int): Number of recurrent layers.
        tpdv (dict): Dictionary for tensor properties.
    """
    
    def __init__(self, config, cent_obs_space, device=torch.device("cuda:0")):
        super(MultiAgentCritic, self).__init__()
        self.hidden_size = config["hidden_size"]
        self._use_orthogonal = config["use_orthogonal"]
        self._use_naive_recurrent_policy = config["use_naive_recurrent_policy"]
        self._use_recurrent_policy = config["use_recurrent_policy"]
        self._recurrent_N = config["recurrent_N"]
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base =  MLPBase
        self.base = base(config, cent_obs_shape)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=0)

        self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        Perform a forward pass through the network to compute value estimates.

        Args:
            cent_obs (torch.Tensor): Centralized observation tensor.
            rnn_states (torch.Tensor): Recurrent states tensor.
            masks (torch.Tensor): Mask tensor.

        Returns:
            tuple: Tuple containing value estimates tensor and new recurrent states tensor.
        """

        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        values = self.v_out(critic_features)

        return values, rnn_states
    