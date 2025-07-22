import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from safepo.common.model import ActorVCritic
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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
    
    # Evaluate start state (s0) performance
    s0_metrics = evaluate_start_state_performance(
        episode_data, returns, create_plots
    )
    
    return {
        'reward_critic': reward_metrics,
        'cost_critic': cost_metrics,
        's0_performance': s0_metrics
    }

def evaluate_start_state_performance(
    episode_data: Dict[str, List[torch.Tensor]],
    returns: Dict[str, List[torch.Tensor]],
    create_plots: bool = False
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate critic performance specifically for start states (s0).
    
    Args:
        episode_data: Dictionary containing lists of tensors from rollout_policy
        returns: Dictionary containing lists of return tensors
        create_plots: Whether to create scatter plots
        
    Returns:
        Dictionary containing evaluation metrics for start state performance
    """
    # Extract start state values and returns
    s0_value_r = []
    s0_value_c = []
    s0_reward_returns = []
    s0_cost_returns = []
    
    for ep_idx in range(len(episode_data['value_r'])):
        # Get first timestep (s0) from each episode
        s0_value_r.append(episode_data['value_r'][ep_idx][0])
        s0_value_c.append(episode_data['value_c'][ep_idx][0])
        s0_reward_returns.append(returns['reward_returns'][ep_idx][0])
        s0_cost_returns.append(returns['cost_returns'][ep_idx][0])
    
    # Stack into tensors
    s0_value_r = torch.stack(s0_value_r)
    s0_value_c = torch.stack(s0_value_c)
    s0_reward_returns = torch.stack(s0_reward_returns)
    s0_cost_returns = torch.stack(s0_cost_returns)
    
    # Evaluate reward critic for s0
    s0_reward_metrics = evaluate_value_estimation_error(
        s0_value_r,
        s0_reward_returns,
        create_plot=create_plots,
        plot_title="Start State (s0) Reward Value Estimates vs MC Returns"
    )
    
    # Evaluate cost critic for s0
    s0_cost_metrics = evaluate_value_estimation_error(
        s0_value_c,
        s0_cost_returns,
        create_plot=create_plots,
        plot_title="Start State (s0) Cost Value Estimates vs MC Returns"
    )
    
    return {
        'reward_critic': s0_reward_metrics,
        'cost_critic': s0_cost_metrics
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