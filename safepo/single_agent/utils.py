import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from safepo.common.model import ActorVCritic
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



def get_activation(name):
    activation_dict = {
        'relu': nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "softmax": nn.Softmax(dim=1),
        "logsoftmax": nn.LogSoftmax(dim=1),
    }

    return activation_dict[name]



def rollout_policy(
    args,
    policy: ActorVCritic,
    env,
    num_episodes: int,
    max_ep_len: int,
    device: torch.device,
    use_risk: bool = False,
    risk_model = None
) -> Dict[str, List[torch.Tensor]]:
    """
    Rollout policy for multiple episodes to collect Monte Carlo returns.
    Assumes vectorized environment.
    
    Args:
        policy: The policy to evaluate
        env: The vectorized environment to rollout in
        num_episodes: Number of episodes to rollout
        max_ep_len: Maximum episode length
        device: Device to run computations on
        use_risk: Whether to use risk estimation
        risk_model: Risk estimation model if use_risk is True
        
    Returns:
        Dictionary containing lists of tensors for:
        - rewards: List of reward sequences
        - costs: List of cost sequences  
        - dones: List of done flags
        - obs: List of observations
        - value_r: List of reward value estimates
        - value_c: List of cost value estimates
    """
    episode_data = {
        'rewards': [],
        'costs': [],
        'dones': [],
        'obs': [],
        'value_r': [],
        'value_c': []
    }
    
    num_envs = env.num_envs
    episodes_completed = 0
    
    while episodes_completed < num_episodes:
        obs, _ = env.reset()
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        
        episode_rewards = []
        episode_costs = []
        episode_dones = []
        episode_obs = []
        episode_value_r = []
        episode_value_c = []
        
        for _ in range(max_ep_len):
            with torch.no_grad():
                risk = torch.exp(risk_model(obs)) if use_risk else None
                _, _, value_r, value_c = policy.step(obs, risk, deterministic=True)
            
            episode_obs.append(obs)
            episode_value_r.append(value_r)
            episode_value_c.append(value_c)
            
            action = policy.actor(obs, risk).sample()
            if "Safe" in args.task:
                next_obs, reward, cost, terminated, truncated, _ = env.step(
                    action.detach().cpu().numpy()
                )
            else:
                next_obs, reward, terminated, truncated, info = env.step(
                    action.detach().cpu().numpy()
                )
                try:
                    cost = info["cost"]
                except:
                    cost = terminated
                
            episode_rewards.append(torch.as_tensor(reward, dtype=torch.float32, device=device))
            episode_costs.append(torch.as_tensor(cost, dtype=torch.float32, device=device))
            episode_dones.append(torch.as_tensor(terminated | truncated, dtype=torch.float32, device=device))
            
            obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
            
            # Check which environments are done
            done_envs = (terminated | truncated).nonzero()[0]
            for env_idx in done_envs:
                if episodes_completed < num_episodes:
                    # Extract and store the episode data for this environment
                    episode_data['rewards'].append(torch.stack([r[env_idx] for r in episode_rewards]))
                    episode_data['costs'].append(torch.stack([c[env_idx] for c in episode_costs]))
                    episode_data['dones'].append(torch.stack([d[env_idx] for d in episode_dones]))
                    episode_data['obs'].append(torch.stack([o[env_idx] for o in episode_obs]))
                    episode_data['value_r'].append(torch.stack([vr[env_idx] for vr in episode_value_r]))
                    episode_data['value_c'].append(torch.stack([vc[env_idx] for vc in episode_value_c]))
                    episodes_completed += 1
            
            if episodes_completed >= num_episodes:
                break
            
            if (terminated | truncated).all():
                break
                
        if episodes_completed >= num_episodes:
            break
            
    return episode_data

def calculate_monte_carlo_returns_from_rollouts(
    episode_data: Dict[str, List[torch.Tensor]],
    gamma: float = 0.99
) -> Dict[str, List[torch.Tensor]]:
    """
    Calculate Monte Carlo returns from rollout data.
    
    Args:
        episode_data: Dictionary containing lists of tensors from rollout_policy
        gamma: Discount factor
        
    Returns:
        Dictionary containing lists of tensors for:
        - reward_returns: List of reward return sequences
        - cost_returns: List of cost return sequences
    """
    returns = {
        'reward_returns': [],
        'cost_returns': []
    }
    
    for ep_idx in range(len(episode_data['rewards'])):
        rewards = episode_data['rewards'][ep_idx]
        costs = episode_data['costs'][ep_idx]
        dones = episode_data['dones'][ep_idx]
        
        seq_len = len(rewards)
        reward_returns = torch.zeros_like(rewards)
        cost_returns = torch.zeros_like(costs)
        
        # Calculate returns from end to start
        for t in range(seq_len-1, -1, -1):
            if t == seq_len-1:
                reward_returns[t] = rewards[t]
                cost_returns[t] = costs[t]
            else:
                reward_returns[t] = rewards[t] + gamma * (1 - dones[t]) * reward_returns[t+1]
                cost_returns[t] = costs[t] + gamma * (1 - dones[t]) * cost_returns[t+1]
        
        returns['reward_returns'].append(reward_returns)
        returns['cost_returns'].append(cost_returns)
    
    return returns

def calculate_correlation(
    value_estimates: torch.Tensor,
    monte_carlo_returns: torch.Tensor
) -> Dict[str, float]:
    """
    Calculate correlation metrics between value estimates and MC returns.
    
    Args:
        value_estimates: Tensor of predicted values
        monte_carlo_returns: Tensor of Monte Carlo returns
        
    Returns:
        Dictionary containing:
        - pearson_corr: Pearson correlation coefficient
        - spearman_corr: Spearman correlation coefficient
        - kendall_corr: Kendall's tau correlation coefficient
    """
    # Convert to numpy for correlation calculations
    values_np = value_estimates.detach().cpu().numpy().flatten()
    returns_np = monte_carlo_returns.detach().cpu().numpy().flatten()
    
    # Calculate correlations
    pearson_corr, _ = stats.pearsonr(values_np, returns_np)
    spearman_corr, _ = stats.spearmanr(values_np, returns_np)
    kendall_corr, _ = stats.kendalltau(values_np, returns_np)
    
    return {
        'pearson_corr': pearson_corr,
        'spearman_corr': spearman_corr,
        'kendall_corr': kendall_corr
    }

def create_value_scatter_plot(
    value_estimates: torch.Tensor,
    monte_carlo_returns: torch.Tensor,
    title: str,
    timesteps: Optional[torch.Tensor] = None
) -> plt.Figure:
    """
    Create a scatter plot comparing value estimates and MC returns.
    
    Args:
        value_estimates: Tensor of predicted values
        monte_carlo_returns: Tensor of Monte Carlo returns
        title: Plot title
        timesteps: Optional tensor of timesteps for coloring points
        
    Returns:
        matplotlib Figure object for wandb logging
    """
    # Convert to numpy
    values_np = value_estimates.detach().cpu().numpy().flatten()
    returns_np = monte_carlo_returns.detach().cpu().numpy().flatten()
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    
    if timesteps is not None:
        # Convert timesteps to numpy and flatten
        timesteps_np = timesteps.detach().cpu().numpy().flatten()
        
        # Create scatter plot with timestep-based coloring
        scatter = plt.scatter(returns_np, values_np, c=timesteps_np, alpha=0.6, cmap='viridis')
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Timestep')
    else:
        # Create scatter plot without coloring
        plt.scatter(returns_np, values_np, alpha=0.2)
    
    # Calculate correlations
    corr = calculate_correlation(value_estimates, monte_carlo_returns)
    
    # Add correlation info to plot
    plt.title(f'{title}\nPearson: {corr["pearson_corr"]:.3f}, Spearman: {corr["spearman_corr"]:.3f}')
    plt.xlabel('Monte Carlo Returns')
    plt.ylabel('Predicted Values')
    
    return fig

def evaluate_value_estimation_error(
    value_estimates: torch.Tensor,
    monte_carlo_returns: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    create_plot: bool = False,
    plot_title: str = "",
    timesteps: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Calculate metrics to evaluate value function estimation bias.
    
    Args:
        value_estimates: Tensor of predicted values
        monte_carlo_returns: Tensor of Monte Carlo returns
        mask: Optional tensor to mask certain timesteps
        create_plot: Whether to create scatter plot
        plot_title: Title for the scatter plot
        timesteps: Optional tensor of timesteps for coloring points
        
    Returns:
        Dictionary containing evaluation metrics and optionally the plot figure
    """
    if mask is not None:
        value_estimates = value_estimates[mask]
        monte_carlo_returns = monte_carlo_returns[mask]
        if timesteps is not None:
            timesteps = timesteps[mask]
    
    # Calculate errors
    errors = value_estimates - monte_carlo_returns
    abs_errors = torch.abs(errors)
    
    # Calculate metrics
    mean_error = errors.mean().item()
    mean_abs_error = abs_errors.mean().item()
    max_error = abs_errors.max().item()
    
    # Calculate over/under estimation ratios
    overestimated = errors > 0
    underestimated = errors < 0
    overestimation_ratio = overestimated.float().mean().item()
    underestimation_ratio = underestimated.float().mean().item()
    
    # Calculate correlations
    correlations = calculate_correlation(value_estimates, monte_carlo_returns)
    
    # Create scatter plot if requested
    plot_fig = None
    if create_plot:
        plot_fig = create_value_scatter_plot(
            value_estimates,
            monte_carlo_returns,
            plot_title,
            timesteps
        )
    
    # Calculate value statistics
    value_stats = {
        'mean_value': value_estimates.mean().item(),
        'std_value': value_estimates.std().item(),
        'min_value': value_estimates.min().item(),
        'max_value': value_estimates.max().item(),
        'mean_mc_return': monte_carlo_returns.mean().item(),
        'std_mc_return': monte_carlo_returns.std().item(),
        'min_mc_return': monte_carlo_returns.min().item(),
        'max_mc_return': monte_carlo_returns.max().item(),
    }
    
    result = {
        'mean_error': mean_error,
        'mean_abs_error': mean_abs_error,
        'overestimation_ratio': overestimation_ratio,
        'underestimation_ratio': underestimation_ratio,
        'max_error': max_error,
        **correlations,
        **value_stats
    }
    
    if create_plot:
        result['plot_fig'] = plot_fig
    
    return result

def evaluate_critic_performance_from_rollouts(
    args,
    policy: ActorVCritic,
    env,
    num_episodes: int,
    max_ep_len: int,
    device: torch.device,
    gamma: float = 0.99,
    use_risk: bool = False,
    risk_model = None,
    create_plots: bool = False
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate critic performance using Monte Carlo returns from fresh rollouts.
    
    Args:
        policy: The policy to evaluate
        env: The environment to rollout in
        num_episodes: Number of episodes to rollout
        max_ep_len: Maximum episode length
        device: Device to run computations on
        gamma: Discount factor
        use_risk: Whether to use risk estimation
        risk_model: Risk estimation model if use_risk is True
        create_plots: Whether to create scatter plots
        
    Returns:
        Dictionary containing evaluation metrics for both reward and cost critics
    """
    # Collect rollout data
    episode_data = rollout_policy(
        args, policy, env, num_episodes, max_ep_len, device, use_risk, risk_model
    )
    
    # Calculate Monte Carlo returns
    returns = calculate_monte_carlo_returns_from_rollouts(episode_data, gamma)
    
    # Flatten all episodes for evaluation
    all_value_r = torch.cat(episode_data['value_r'])
    all_value_c = torch.cat(episode_data['value_c'])
    all_reward_returns = torch.cat(returns['reward_returns'])
    all_cost_returns = torch.cat(returns['cost_returns'])
    
    # Create timestep information for coloring
    timesteps_list = []
    for ep_idx, episode_values in enumerate(episode_data['value_r']):
        seq_len = len(episode_values)
        episode_timesteps = torch.arange(seq_len, device=device)
        timesteps_list.append(episode_timesteps)
    all_timesteps = torch.cat(timesteps_list)
    
    # Evaluate reward critic
    reward_metrics = evaluate_value_estimation_error(
        all_value_r,
        all_reward_returns,
        create_plot=create_plots,
        plot_title="Reward Value Estimates vs MC Returns",
        timesteps=all_timesteps
    )
    
    # Evaluate cost critic
    cost_metrics = evaluate_value_estimation_error(
        all_value_c,
        all_cost_returns,
        create_plot=create_plots,
        plot_title="Cost Value Estimates vs MC Returns",
        timesteps=all_timesteps
    )
    
    return {
        'reward_critic': reward_metrics,
        'cost_critic': cost_metrics
    }

# Keep the original functions for backward compatibility
def calculate_monte_carlo_returns(
    rewards: torch.Tensor,
    costs: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate Monte Carlo returns for both reward and cost.
    
    Args:
        rewards: Tensor of shape (batch_size, sequence_length) containing rewards
        costs: Tensor of shape (batch_size, sequence_length) containing costs
        dones: Tensor of shape (batch_size, sequence_length) containing done flags
        gamma: Discount factor
        
    Returns:
        Tuple of (reward_returns, cost_returns) tensors of same shape as input
    """
    batch_size, seq_len = rewards.shape
    reward_returns = torch.zeros_like(rewards)
    cost_returns = torch.zeros_like(costs)
    
    # Calculate returns from end to start
    for t in range(seq_len-1, -1, -1):
        if t == seq_len-1:
            reward_returns[:, t] = rewards[:, t]
            cost_returns[:, t] = costs[:, t]
        else:
            reward_returns[:, t] = rewards[:, t] + gamma * (1 - dones[:, t]) * reward_returns[:, t+1]
            cost_returns[:, t] = costs[:, t] + gamma * (1 - dones[:, t]) * cost_returns[:, t+1]
            
    return reward_returns, cost_returns

def evaluate_critic_performance(
    buffer_data: Dict[str, torch.Tensor],
    gamma: float = 0.99,
    create_plots: bool = False
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate both reward and cost critic performance using Monte Carlo returns.
    
    Args:
        buffer_data: Dictionary containing buffer data with keys:
            - 'reward': Tensor of rewards
            - 'cost': Tensor of costs
            - 'done': Tensor of done flags
            - 'value_r': Tensor of reward value estimates
            - 'value_c': Tensor of cost value estimates
        gamma: Discount factor
        create_plots: Whether to create scatter plots with timestep coloring
        
    Returns:
        Dictionary containing evaluation metrics for both reward and cost critics
    """
    # Calculate Monte Carlo returns
    reward_returns, cost_returns = calculate_monte_carlo_returns(
        buffer_data['reward'],
        buffer_data['cost'],
        buffer_data['done'],
        gamma
    )
    
    # Create timestep information for coloring if plots are requested
    timesteps = None
    if create_plots:
        batch_size, seq_len = buffer_data['reward'].shape
        timesteps = torch.arange(seq_len, device=buffer_data['reward'].device).unsqueeze(0).expand(batch_size, seq_len)
    
    # Evaluate reward critic
    reward_metrics = evaluate_value_estimation_error(
        buffer_data['value_r'],
        reward_returns,
        create_plot=create_plots,
        plot_title="Reward Value Estimates vs MC Returns",
        timesteps=timesteps
    )
    
    # Evaluate cost critic
    cost_metrics = evaluate_value_estimation_error(
        buffer_data['value_c'],
        cost_returns,
        create_plot=create_plots,
        plot_title="Cost Value Estimates vs MC Returns",
        timesteps=timesteps
    )
    
    return {
        'reward_critic': reward_metrics,
        'cost_critic': cost_metrics
    } 
import os
import pickle
import torch
import numpy as np
from random import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import tqdm

def train_risk(model, dataloader, criterion, opt, num_epochs, device):
    model.train()
    net_loss = 0
    for _ in tqdm.tqdm(range(num_epochs)):
        for batch in dataloader:
                pred = model(batch[0].to(device))
                loss = criterion(pred, torch.argmax(batch[1].squeeze(), axis=1).to(device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                net_loss += loss.item()
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "risk_model.pt"))
    wandb.save("risk_model.pt")
    model.eval()
    return net_loss





def make_dirs(traj_path, episode):
        #try:
        os.makedirs(os.path.join(traj_path, "traj_%d"%episode, "lidar"))
        os.makedirs(os.path.join(traj_path, "traj_%d"%episode, "info"))
        
        #except:
        #    pass


def compute_fear(costs, max_dist=1000):
        fear_fwd, fear_bwd = torch.full(costs.size(), max_dist), torch.full(costs.size(), max_dist)
        fwd_flag, bwd_flag = 0, 0
        fwd_counter, bwd_counter = 0, 0
        len_run = len(costs)
        for i in range(len_run):
                if costs[i] == 1:
                        fear_fwd[i] = 0
                        fwd_flag = 1
                        fwd_counter = 0
                elif fwd_flag:
                       fwd_counter += 1
                       fear_fwd[i] = fwd_counter

                if costs[len_run-i-1] == 1:
                        bwd_flag = 1
                        fear_bwd[len_run-i-1] = 0
                        bwd_counter = 0
                elif bwd_flag:
                       bwd_counter += 1
                       fear_bwd[len_run-i-1] = bwd_counter
        return torch.min(fear_fwd, fear_bwd)

                     


def store_data(next_obs, info_dict, traj_path, episode, step_log):
        #, 'prev_obs_rgb': obs['vision']}
        #info_dict.update(obs)
        ## Saving the info for this step
        f1 = open(os.path.join(traj_path, "traj_%d"%episode, "info", "%d.pkl"%step_log), "wb")
        pickle.dump(info_dict, f1, protocol=pickle.HIGHEST_PROTOCOL)
        f1.close()
        # del obs['vision']
        ## Saving data from other sensors (particularly lidar)
        f2 = open(os.path.join(traj_path, "traj_%d"%episode, "lidar", "%d.pkl"%step_log), "wb")
        pickle.dump(next_obs, f2, protocol=pickle.HIGHEST_PROTOCOL)
        f2.close()


def get_activation(name):
    activation_dict = {
        'relu': nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "softmax": nn.Softmax(dim=1),
        "logsoftmax": nn.LogSoftmax(dim=1),
    }

    return activation_dict[name]



def make_state_action_risk_data(data_path):
        obs = torch.load(os.path.join(data_path, "obs.pt"))
        actions = torch.load(os.path.join(data_path, "actions.pt"))
        risks = torch.load(os.path.join(data_path, "risks.pt"))
        ep_len = torch.load(os.path.join(data_path, "ep_len.pt"))
        state_action_risk_data = None
        for idx in range(1, len(ep_len)):
                start, end = int(ep_len[idx-1]), int(ep_len[idx])
                print(start, end)
                obs_idx = obs[start:end]
                actions_idx = actions[start:end]
                risks_idx = risks[start:end]
                print(obs_idx.size(), actions_idx.size(), risks_idx.size())
                sar_data = torch.cat([obs_idx[:-1], actions_idx[1:], risks_idx[1:]], axis=1)
                state_action_risk_data = sar_data if state_action_risk_data is None else torch.cat([state_action_risk_data, sar_data], axis=0)
        torch.save(state_action_risk_data, os.path.join(data_path, "state_action_risk.pt"))
        return state_action_risk_data

def make_state_risk_data(data_path):
        obs = torch.load(os.path.join(data_path, "obs.pt"))
        risks = torch.load(os.path.join(data_path, "risks.pt"))
        ep_len = torch.load(os.path.join(data_path, "ep_len.pt"))
        return torch.cat([obs, risks], axis=1)

def combine_data(data_path, type="state_risk"):
        for env in os.listdir(data_path):
                env_path = os.path.join(data_path, env)
                all_data = None
                for run in os.listdir(env_path):
                        run_path = os.path.join(env_path, run)
                        if type == "state_risk":
                                try:
                                        data = make_state_risk_data(run_path)
                                except:
                                        pass
                        else:
                                try:
                                        data = make_state_action_risk_data(run_path)
                                except:
                                        pass
                all_data = data if all_data is None else torch.cat([all_data, data], axis=0)
        torch.save(all_data, os.path.join(env_path, "all_%s.pt"%type))



class ReplayBuffer:
        def __init__(self, buffer_size, obs_dim, risk_size, device):
                self.obs = None
                self.next_obs = torch.zeros(buffer_size, obs_dim).to(device)
                self.actions = None
                self.rewards = None
                self.dones = None
                self.risks = torch.zeros(buffer_size, risk_size).to(device)
                self.dist_to_fails = torch.zeros(buffer_size, 1).to(device)
                self.costs = None
                #self.data_path = data_path
                self.buffer_size = buffer_size
                self.buff_fill = 0

        def add(self, obs, next_obs, action, reward, done, cost, risk, dist_to_fail):
                data_size = next_obs.size()[0]
                self.next_obs[self.buff_fill:self.buff_fill+data_size, :] = next_obs.squeeze()
                self.risks[self.buff_fill:self.buff_fill+data_size, :] = risk.squeeze()
                self.dist_to_fails[self.buff_fill:self.buff_fill+data_size, :] = dist_to_fail.reshape(-1, 1)
                self.buff_fill += data_size

                #self.obs = obs if self.obs is None else torch.concat([self.obs, obs], axis=0)
                #self.next_obs = next_obs if self.next_obs is None else torch.concat([self.next_obs, next_obs], axis=0)
                #self.actions = action if self.actions is None else torch.concat([self.actions, action], axis=0)
                #self.rewards = reward if self.rewards is None else torch.concat([self.rewards, reward], axis=0)
                #self.dones = done if self.dones is None else torch.concat([self.dones, done], axis=0)
                #self.risks = risk if self.risks is None else torch.concat([self.risks, risk], axis=0)
                #self.costs = cost if self.costs is None else torch.concat([self.costs, cost], axis=0)
                #self.dist_to_fails = dist_to_fail if self.dist_to_fails is None else torch.concat([self.dist_to_fails, dist_to_fail], axis=0)

        def __len__(self):
            return self.buff_fill

        def sample(self, sample_size):
                #if self.next_obs.size()[0] > self.buffer_size:
                #    self.next_obs = self.next_obs[-self.buffer_size:]
                #    self.risks = self.risks[-self.buffer_size:]
                sample_idx = np.random.randint(1, self.buff_fill, size=sample_size)
                return {"obs": None, #self.obs[sample_idx],
                        "next_obs": self.next_obs[sample_idx],
                        "actions": None, #self.actions[sample_idx],
                        "rewards": None, #self.rewards[sample_idx],
                        "dones": None, #self.dones[sample_idx],
                        "risks": self.risks[sample_idx],
                        "costs": None, #self.costs[sample_idx],
                        "dist_to_fail": self.dist_to_fails[sample_idx]}
        
        def sample_balanced(self, sample_size):
                idx = range(self.obs.size()[0])
                print(self.risks.size())
                
                idx_risky = idx[torch.argmax(self.risks, 1).squeeze().cpu().numpy() == 1]
                idx_safe  = idx[torch.argmax(self.risks, 1).squeeze().cpu().numpy() == 0]
                sample_idx = np.array(list(np.random.choice(idx_risky, sample_size/2)) + list(np.random.choice(idx_safe, sample_size/2)))
                return {"obs": self.obs[sample_idx],
                        "next_obs": self.next_obs[sample_idx],
                        "actions": self.actions[sample_idx],
                        "rewards": self.rewards[sample_idx],
                        "dones": self.dones[sample_idx],
                        "risks": self.risks[sample_idx], 
                        "costs": self.costs[sample_idx],
                        "dist_to_fail": self.dist_to_fails[sample_idx]}
                  

        def slice_data(self, min_idx, max_idx):
                idx = range(min_idx, max_idx)
                sample_idx = idx #np.random.choice(idx, sample_size)
                return {"obs": self.obs[sample_idx],
                        "next_obs": self.next_obs[sample_idx],
                        "actions": self.actions[sample_idx],
                        "rewards": self.rewards[sample_idx],
                        "dones": self.dones[sample_idx],
                        "risks": self.risks[sample_idx], 
                        "costs": self.costs[sample_idx],
                        "dist_to_fail": self.dist_to_fails[sample_idx]}        

        def save(self):
            torch.save(self.next_obs, os.path.join(self.data_path, "all_obs.pt"))
            torch.save(self.risks, os.path.join(self.data_path, "all_risks.pt"))



class ReplayBufferBalanced:
        def __init__(self, buffer_size=100000):
                self.obs_risky = None 
                self.next_obs_risky = None
                self.actions_risky = None 
                self.rewards_risky = None 
                self.dones_risky = None
                self.risks_risky = None 
                self.dist_to_fails_risky = None 
                self.costs_risky = None

                self.obs_safe = None 
                self.next_obs_safe = None
                self.actions_safe = None 
                self.rewards_safe = None 
                self.dones_safe = None
                self.risks_safe = None 
                self.dist_to_fails_safe = None 
                self.costs_safe = None

        def add_risky(self, obs, next_obs, action, reward, done, cost, risk, dist_to_fail):
                self.obs_risky = obs if self.obs_risky is None else torch.concat([self.obs_risky, obs], axis=0)
                self.next_obs_risky = next_obs if self.next_obs_risky is None else torch.concat([self.next_obs_risky, next_obs], axis=0)
                self.actions_risky = action if self.actions_risky is None else torch.concat([self.actions_risky, action], axis=0)
                self.rewards_risky = reward if self.rewards_risky is None else torch.concat([self.rewards_risky, reward], axis=0)
                self.dones_risky = done if self.dones_risky is None else torch.concat([self.dones_risky, done], axis=0)
                self.risks_risky = risk if self.risks_risky is None else torch.concat([self.risks_risky, risk], axis=0)
                self.costs_risky = cost if self.costs_risky is None else torch.concat([self.costs_risky, cost], axis=0)
                self.dist_to_fails_risky = dist_to_fail if self.dist_to_fails_risky is None else torch.concat([self.dist_to_fails_risky, dist_to_fail], axis=0)

        def add_safe(self, obs, next_obs, action, reward, done, cost, risk, dist_to_fail):
                self.obs_safe = obs if self.obs_safe is None else torch.concat([self.obs_safe, obs], axis=0)
                self.next_obs_safe = next_obs if self.next_obs_safe is None else torch.concat([self.next_obs_safe, next_obs], axis=0)
                self.actions_safe = action if self.actions_safe is None else torch.concat([self.actions_safe, action], axis=0)
                self.rewards_safe = reward if self.rewards_safe is None else torch.concat([self.rewards_safe, reward], axis=0)
                self.dones_safe = done if self.dones_safe is None else torch.concat([self.dones_safe, done], axis=0)
                self.risks_safe = risk if self.risks_safe is None else torch.concat([self.risks_safe, risk], axis=0)
                self.costs_safe = cost if self.costs_safe is None else torch.concat([self.costs_safe, cost], axis=0)
                self.dist_to_fails_safe = dist_to_fail if self.dist_to_fails_safe is None else torch.concat([self.dist_to_fails_safe, dist_to_fail], axis=0)

        
        def sample(self, sample_size):
                idx_risky = range(self.obs_risky.size()[0])
                idx_safe = range(self.obs_safe.size()[0])

                sample_risky_idx = np.random.choice(idx_risky, int(sample_size/2))
                sample_safe_idx = np.random.choice(idx_safe, int(sample_size/2))

                return {"obs": torch.cat([self.obs_risky[sample_risky_idx], self.obs_safe[sample_safe_idx]], 0),
                        "next_obs": torch.cat([self.next_obs_risky[sample_risky_idx], self.next_obs_safe[sample_safe_idx]], 0),
                        "actions": torch.cat([self.actions_risky[sample_risky_idx], self.actions_safe[sample_safe_idx]], 0),
                        "rewards": torch.cat([self.rewards_risky[sample_risky_idx], self.rewards_safe[sample_safe_idx]], 0),
                        "dones": torch.cat([self.dones_risky[sample_risky_idx], self.dones_safe[sample_safe_idx]], 0),
                        "risks": torch.cat([self.risks_risky[sample_risky_idx], self.risks_safe[sample_safe_idx]], 0),
                        "costs": torch.cat([self.costs_risky[sample_risky_idx], self.costs_safe[sample_safe_idx]], 0),
                        "dist_to_fail": torch.cat([self.dist_to_fails_risky[sample_risky_idx], self.dist_to_fails_safe[sample_safe_idx]], 0),}
        



                        

                


class BayesRiskEstCont(nn.Module):
    def __init__(self, obs_size=64, fc1_size=128, fc2_size=128, fc3_size=128, fc4_size=128, out_size=1, model_type="state_risk", action_size=2):
        super().__init__()
        self.obs_size = obs_size
        self.model_type = model_type
        self.action_size = action_size

        self.fc1 = nn.Linear(obs_size, fc1_size)
        if self.model_type == "state_risk":
            self.fc2 = nn.Linear(fc1_size, fc2_size)
        else:
            self.fc1_action = nn.Linear(action_size, int(fc1_size/2))
            self.fc2 = nn.Linear(fc1_size + int(fc1_size/2), fc2_size)
            self.bnorm1_action = nn.BatchNorm1d(int(fc1_size/2))

        self.mean_fc3 = nn.Linear(fc2_size, fc3_size)
        self.mean_fc4 = nn.Linear(fc3_size, fc4_size)
        self.mean_out = nn.Linear(fc4_size, out_size)

        self.logvar_fc3 = nn.Linear(fc2_size, fc3_size)
        self.logvar_fc4 = nn.Linear(fc3_size, fc4_size)
        self.logvar_out = nn.Linear(fc4_size, out_size)


        ## Batch Norm layers
        self.bnorm1 = nn.BatchNorm1d(fc1_size)
        self.bnorm2 = nn.BatchNorm1d(fc2_size)
        self.mean_bnorm3 = nn.BatchNorm1d(fc3_size)
        self.mean_bnorm4 = nn.BatchNorm1d(fc4_size)

        #self.var_bnorm1 = nn.BatchNorm1d(fc1_size)
        #self.var_bnorm2 = nn.BatchNorm1d(fc2_size)
        self.var_bnorm3 = nn.BatchNorm1d(fc3_size)
        self.var_bnorm4 = nn.BatchNorm1d(fc4_size)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.2)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, action=None):
        x = self.bnorm1(self.relu(self.fc1(x)))
        if self.model_type == "state_action_risk":
            x1 = self.bnorm1_action(self.relu(self.fc1_action(action)))
            x = torch.cat([x, x1], axis=1)

        x = self.bnorm2(self.relu(self.fc2(x)))

        mean  = self.mean_bnorm3(self.relu(self.mean_fc3(x)))
        mean  = self.mean_bnorm4(self.relu(self.mean_fc4(mean)))
        mean  = self.sigmoid(self.mean_out(mean))

        logvar = self.var_bnorm3(self.relu(self.logvar_fc3(x)))
        logvar = self.var_bnorm4(self.relu(self.logvar_fc4(x)))
        logvar = self.sigmoid(self.logvar_out(x))

        #x = self.bnorm3(self.relu(self.dropout(self.fc3(x))))
        #x = self.bnorm4(self.relu(self.dropout(self.fc4(x))))
        #out = self.logsoftmax(self.out(x))
        return mean, logvar



class BayesRiskEst(nn.Module):
    def __init__(self, obs_size=64, fc1_size=64, fc2_size=64,\
                  fc3_size=64, fc4_size=64, out_size=2, batch_norm=True, activation='relu', model_type="state_risk", action_size=2):
        super().__init__()
        self.obs_size = obs_size
        self.batch_norm = batch_norm
        self.model_type = model_type
        self.fc1 = nn.Linear(obs_size, fc1_size)
        if self.model_type == "state_risk":
            self.fc2 = nn.Linear(fc1_size, fc2_size)
        else:
            self.fc1_action = nn.Linear(action_size, int(fc1_size/2))
            self.fc2 = nn.Linear(fc1_size + int(fc1_size/2), fc2_size)
            self.bnorm1_action = nn.BatchNorm1d(int(fc1_size/2))

        #self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.fc4 = nn.Linear(fc3_size, fc4_size)
        self.out = nn.Linear(fc4_size, out_size)

        ## Batch Norm layers
        self.bnorm1 = nn.BatchNorm1d(fc1_size)
        self.bnorm2 = nn.BatchNorm1d(fc2_size)
        self.bnorm3 = nn.BatchNorm1d(fc3_size)
        self.bnorm4 = nn.BatchNorm1d(fc4_size)

        # Activation functions
        self.activation = get_activation(activation)

        self.logsoftmax = get_activation("logsoftmax")
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, action=None):
        # Taking care of any augmentation in the observation space
        x = x[:, :self.obs_size]
        if self.batch_norm:
            x = self.bnorm1(self.activation(self.fc1(x)))
            if self.model_type == "state_action_risk":
                x1 = self.bnorm1_action(self.activation(self.fc1_action(action)))
                x = torch.cat([x, x1], axis=1)
            #x = self.bnorm2(self.activation(self.fc2(x)))
            # x = self.bnorm3(self.activation(self.dropout(self.fc3(x))))
            x = self.bnorm4(self.activation(self.dropout(self.fc4(x))))
        else:
            x = self.activation(self.fc1(x))
            if self.model_type == "state_action_risk":
                x1 = self.activation(self.fc1_action(action))
                x = torch.cat([x, x1], axis=1)

            #x = self.activation(self.fc2(x))
            # x = self.activation(self.dropout(self.fc3(x)))
            x = self.activation(self.dropout(self.fc4(x)))

        out = self.logsoftmax(self.out(x))
        return out


class RiskEst(nn.Module):
    def __init__(self, obs_size=64, fc1_size=128, fc2_size=128,\
                  fc3_size=128, fc4_size=128, out_size=2, batch_norm=False, activation='relu', continuous_risk=False):
        super().__init__()
        self.obs_size = obs_size
        self.batch_norm = batch_norm
        self.continuous_risk = continuous_risk

        self.fc1 = nn.Linear(obs_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.fc4 = nn.Linear(fc3_size, fc4_size)
        self.out = nn.Linear(fc4_size, out_size)

        ## Batch Norm layers
        self.bnorm1 = nn.BatchNorm1d(fc1_size)
        self.bnorm2 = nn.BatchNorm1d(fc2_size)
        self.bnorm3 = nn.BatchNorm1d(fc3_size)
        self.bnorm4 = nn.BatchNorm1d(fc4_size)

        # Activation functions
        self.activation = get_activation(activation)
        self.softmax = get_activation("softmax")

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        if self.batch_norm:
            x = self.bnorm1(self.activation(self.fc1(x)))
            x = self.bnorm2(self.activation(self.fc2(x)))
            x = self.bnorm3(self.activation(self.dropout(self.fc3(x))))
            x = self.bnorm4(self.activation(self.dropout(self.fc4(x))))
        else:
            x = self.activation(self.fc1(x))
            x = self.activation(self.fc2(x))
            x = self.activation(self.dropout(self.fc3(x)))
            x = self.activation(self.dropout(self.fc4(x)))    
        
        if self.continuous_risk:
            out = self.sigmoid(self.out(x))
        else:
            out = self.softmax(self.out(x))
        return out