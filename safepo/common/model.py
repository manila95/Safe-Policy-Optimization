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
from torch.distributions import Normal
from safepo.utils.act import ACTLayer
from safepo.utils.mlp import MLPBase
from safepo.utils.util import check, init
from safepo.utils.util import get_shape_from_obs_space


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


class QuantileCritic(nn.Module):
    """
    Quantile critic network for distributional value estimation.

    This class represents a critic network that estimates multiple quantiles of the value distribution.

    Args:
        obs_dim (int): Dimensionality of the observation space.
        n_quantiles (int): Number of quantiles to estimate (default: 25).
        hidden_sizes (list): List of hidden layer sizes.
        use_risk (bool): Whether to use risk information.
        risk_size (int): Size of risk information.
        use_layer_norm (bool): Whether to use layer normalization.

    Attributes:
        n_quantiles (int): Number of quantiles.
        critic (nn.Sequential): MLP network outputting quantile values.

    Example:
        obs_dim = 10
        critic = QuantileCritic(obs_dim, n_quantiles=25)
        observation = torch.randn(1, obs_dim)
        quantiles = critic(observation)  # Shape: [batch_size, n_quantiles]
    """

    def __init__(self, obs_dim, n_quantiles: int = 25, hidden_sizes: list = [64, 64], 
                 use_risk=False, risk_size=None, use_layer_norm=False):
        super().__init__()
        self.use_risk = use_risk
        self.n_quantiles = n_quantiles
        
        if self.use_risk:
            self.critic = build_risk_mlp_network([obs_dim] + hidden_sizes + [n_quantiles], risk_size, use_layer_norm=use_layer_norm)
        else:
            self.critic = build_mlp_network([obs_dim] + hidden_sizes + [n_quantiles], use_layer_norm=use_layer_norm)

    def forward(self, obs, risk=None):
        """
        Forward pass through the quantile critic.

        Args:
            obs (torch.Tensor): Input observation tensor.
            risk (torch.Tensor, optional): Risk information tensor.

        Returns:
            torch.Tensor: Quantile values of shape [batch_size, n_quantiles].
        """
        if self.use_risk:
            return self.critic(obs, risk)
        else:
            return self.critic(obs)


class TruncatedQuantileCritic(nn.Module):
    """
    Truncated Quantile Critic for value estimation with reduced overestimation bias.

    This class implements the TQC (Truncated Quantile Critics) approach, using multiple
    quantile networks to estimate the value distribution and truncating across predictions
    to reduce overestimation bias.

    Args:
        obs_dim (int): Dimensionality of the observation space.
        n_critics (int): Number of critic networks (default: 5).
        n_quantiles (int): Number of quantiles per critic (default: 25).
        n_truncate (int): Number of quantiles to keep after truncation (default: 2).
                         If None, uses min(n_critics * n_quantiles // 2, n_critics * n_quantiles - 1).
        hidden_sizes (list): List of hidden layer sizes.
        use_risk (bool): Whether to use risk information.
        risk_size (int): Size of risk information.
        use_layer_norm (bool): Whether to use layer normalization.
        huber_kappa (float): Threshold for Huber loss (default: 1.0).

    Attributes:
        n_critics (int): Number of critic networks.
        n_quantiles (int): Number of quantiles per critic.
        n_truncate (int): Number of quantiles to keep after truncation.
        critics (nn.ModuleList): List of quantile critic networks.
        huber_kappa (float): Threshold for Huber loss.

    Example:
        obs_dim = 10
        critic = TruncatedQuantileCritic(obs_dim, n_critics=5, n_quantiles=25, n_truncate=2)
        observation = torch.randn(8, obs_dim)
        
        # Get value estimate (mean of truncated quantiles)
        value = critic.get_value(observation)
        
        # Get all quantiles for loss computation
        all_quantiles = critic(observation)  # Shape: [batch_size, n_critics, n_quantiles]
    """

    def __init__(self, obs_dim, n_critics: int = 5, n_quantiles: int = 25, n_truncate: int = None,
                 hidden_sizes: list = [64, 64], use_risk=False, risk_size=None, 
                 use_layer_norm=False, huber_kappa: float = 1.0):
        super().__init__()
        self.n_critics = n_critics
        self.n_quantiles = n_quantiles
        self.use_risk = use_risk
        self.huber_kappa = huber_kappa
        
        # Default truncation: keep roughly half of all quantiles
        if n_truncate is None:
            total_quantiles = n_critics * n_quantiles
            self.n_truncate = min(total_quantiles // 2, total_quantiles - 1)
        else:
            self.n_truncate = n_truncate
            
        # Create multiple quantile critics
        self.critics = nn.ModuleList([
            QuantileCritic(obs_dim, n_quantiles, hidden_sizes, use_risk, risk_size, use_layer_norm)
            for _ in range(n_critics)
        ])

    def forward(self, obs, risk=None):
        """
        Forward pass through all quantile critics.

        Args:
            obs (torch.Tensor): Input observation tensor of shape [batch_size, obs_dim].
            risk (torch.Tensor, optional): Risk information tensor.

        Returns:
            torch.Tensor: All quantile values of shape [batch_size, n_critics, n_quantiles].
        """
        if self.use_risk:
            quantiles = [critic(obs, risk) for critic in self.critics]
        else:
            quantiles = [critic(obs) for critic in self.critics]
        
        # Stack to shape [batch_size, n_critics, n_quantiles]
        return torch.stack(quantiles, dim=1)

    def get_value(self, obs, risk=None):
        """
        Get the value estimate by truncating and averaging quantiles.

        Args:
            obs (torch.Tensor): Input observation tensor of shape [batch_size, obs_dim].
            risk (torch.Tensor, optional): Risk information tensor.

        Returns:
            torch.Tensor: Value estimate of shape [batch_size].
        """
        with torch.no_grad():
            all_quantiles = self.forward(obs, risk)  # [batch_size, n_critics, n_quantiles]
            batch_size = all_quantiles.shape[0]
            
            # Flatten quantiles across critics: [batch_size, n_critics * n_quantiles]
            all_quantiles = all_quantiles.view(batch_size, -1)
            
            # Sort and truncate: keep the smallest n_truncate quantiles
            sorted_quantiles, _ = torch.sort(all_quantiles, dim=1)
            truncated_quantiles = sorted_quantiles[:, :self.n_truncate]
            
            # Return mean of truncated quantiles
            return truncated_quantiles.mean(dim=1)

    def quantile_huber_loss(self, quantiles, targets):
        """
        Compute the quantile Huber loss for distributional RL.

        Args:
            quantiles (torch.Tensor): Predicted quantiles of shape [batch_size, n_quantiles].
            targets (torch.Tensor): Target values of shape [batch_size].

        Returns:
            torch.Tensor: Quantile Huber loss value.
        """
        batch_size = quantiles.shape[0]
        
        # Expand targets to match quantiles shape
        targets = targets.unsqueeze(-1)  # [batch_size, 1]
        
        # Compute TD errors
        td_errors = targets - quantiles  # [batch_size, n_quantiles]
        
        # Huber loss
        huber_loss = torch.where(
            td_errors.abs() <= self.huber_kappa,
            0.5 * td_errors.pow(2),
            self.huber_kappa * (td_errors.abs() - 0.5 * self.huber_kappa)
        )
        
        # Quantile weights (tau values)
        tau = torch.arange(0.5 / self.n_quantiles, 1.0, 1.0 / self.n_quantiles, 
                          device=quantiles.device, dtype=quantiles.dtype)
        tau = tau.view(1, -1)  # [1, n_quantiles]
        
        # Quantile regression loss
        quantile_weight = torch.abs(tau - (td_errors < 0).float())
        quantile_loss = quantile_weight * huber_loss
        
        return quantile_loss.mean()

    def compute_loss(self, obs, targets, risk=None):
        """
        Compute the total loss across all quantile critics.

        Args:
            obs (torch.Tensor): Input observation tensor of shape [batch_size, obs_dim].
            targets (torch.Tensor): Target values of shape [batch_size].
            risk (torch.Tensor, optional): Risk information tensor.

        Returns:
            torch.Tensor: Total loss value.
        """
        all_quantiles = self.forward(obs, risk)  # [batch_size, n_critics, n_quantiles]
        
        total_loss = 0.0
        for i in range(self.n_critics):
            quantiles = all_quantiles[:, i, :]  # [batch_size, n_quantiles]
            loss = self.quantile_huber_loss(quantiles, targets)
            total_loss += loss
            
        return total_loss / self.n_critics


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

    Example:
        obs_dim = 10
        act_dim = 2
        actor_critic = ActorVCritic(obs_dim, act_dim)
        observation = torch.randn(1, obs_dim)
        action, log_prob, reward_value, cost_value = actor_critic.step(observation)
        value_estimate = actor_critic.get_value(observation)
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes: list = [64, 64], use_risk=False, risk_size=None, 
                 use_actor_layer_norm=False, use_critic_layer_norm=False):
        super().__init__()
        self.use_risk = use_risk
        self.reward_critic = VCritic(obs_dim, hidden_sizes, use_risk=use_risk, risk_size=risk_size, 
                                    use_layer_norm=use_critic_layer_norm)
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
        Take a policy step based on observations.

        Args:
            obs (torch.Tensor): Input observation tensor.
            deterministic (bool): Flag indicating whether to take a deterministic action.

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
            value_c = self.cost_critic(obs, risk)
        else:
            value_r = self.reward_critic(obs)
            value_c = self.cost_critic(obs)
        return action, log_prob, value_r, value_c


class ActorTQC(nn.Module):
    """
    Actor-critic policy with Truncated Quantile Critics.

    This class combines an actor network with truncated quantile critics for both reward
    and cost estimation, providing better value estimates with reduced overestimation bias.

    Args:
        obs_dim (int): Dimensionality of the observation space.
        act_dim (int): Dimensionality of the action space.
        hidden_sizes (list): List of hidden layer sizes.
        n_critics (int): Number of critic networks per TQC (default: 5).
        n_quantiles (int): Number of quantiles per critic (default: 25).
        n_truncate_reward (int): Number of quantiles to keep for reward (default: None).
        n_truncate_cost (int): Number of quantiles to keep for cost (default: None).
        use_risk (bool): Whether to use risk information.
        risk_size (int): Size of risk information.
        use_actor_layer_norm (bool): Whether to use layer normalization in actor.
        use_critic_layer_norm (bool): Whether to use layer normalization in critics.
        huber_kappa (float): Threshold for Huber loss (default: 1.0).

    Example:
        obs_dim = 10
        act_dim = 2
        actor_critic = ActorTQC(obs_dim, act_dim, n_critics=5, n_quantiles=25)
        observation = torch.randn(1, obs_dim)
        action, log_prob, reward_value, cost_value = actor_critic.step(observation)
        
        # For training, compute losses:
        reward_targets = torch.randn(32)
        cost_targets = torch.randn(32)
        reward_loss = actor_critic.reward_critic.compute_loss(observation, reward_targets)
        cost_loss = actor_critic.cost_critic.compute_loss(observation, cost_targets)
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes: list = [64, 64], 
                 n_critics: int = 5, n_quantiles: int = 25,
                 n_truncate_reward: int = None, n_truncate_cost: int = None,
                 use_risk=False, risk_size=None, 
                 use_actor_layer_norm=False, use_critic_layer_norm=False,
                 huber_kappa: float = 1.0):
        super().__init__()
        self.use_risk = use_risk
        
        # Create truncated quantile critics for reward and cost
        self.reward_critic = TruncatedQuantileCritic(
            obs_dim=obs_dim,
            n_critics=n_critics,
            n_quantiles=n_quantiles,
            n_truncate=n_truncate_reward,
            hidden_sizes=hidden_sizes,
            use_risk=use_risk,
            risk_size=risk_size,
            use_layer_norm=use_critic_layer_norm,
            huber_kappa=huber_kappa
        )
        
        self.cost_critic = TruncatedQuantileCritic(
            obs_dim=obs_dim,
            n_critics=n_critics,
            n_quantiles=n_quantiles,
            n_truncate=n_truncate_cost,
            hidden_sizes=hidden_sizes,
            use_risk=use_risk,
            risk_size=risk_size,
            use_layer_norm=use_critic_layer_norm,
            huber_kappa=huber_kappa
        )
        
        # Actor network
        self.actor = Actor(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes,
            use_risk=use_risk,
            risk_size=risk_size,
            use_layer_norm=use_actor_layer_norm
        )

    def get_value(self, obs, risk=None):
        """
        Estimate the reward value of observations using the truncated quantile critic.

        Args:
            obs (torch.Tensor): Input observation tensor.
            risk (torch.Tensor, optional): Risk information tensor.

        Returns:
            torch.Tensor: Estimated reward value for the input observation.
        """
        if self.use_risk:
            return self.reward_critic.get_value(obs, risk)
        else:
            return self.reward_critic.get_value(obs)

    def get_cost_value(self, obs, risk=None):
        """
        Estimate the cost value of observations using the truncated quantile critic.

        Args:
            obs (torch.Tensor): Input observation tensor.
            risk (torch.Tensor, optional): Risk information tensor.

        Returns:
            torch.Tensor: Estimated cost value for the input observation.
        """
        if self.use_risk:
            return self.cost_critic.get_value(obs, risk)
        else:
            return self.cost_critic.get_value(obs)

    def step(self, obs, risk=None, deterministic=False):
        """
        Take a policy step based on observations.

        Args:
            obs (torch.Tensor): Input observation tensor.
            risk (torch.Tensor, optional): Risk information tensor.
            deterministic (bool): Flag indicating whether to take a deterministic action.

        Returns:
            tuple: Tuple containing action tensor, log probabilities of the action,
                   reward value estimate, and cost value estimate.
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
            value_r = self.reward_critic.get_value(obs, risk)
            value_c = self.cost_critic.get_value(obs, risk)
        else:
            value_r = self.reward_critic.get_value(obs)
            value_c = self.cost_critic.get_value(obs)
            
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
    