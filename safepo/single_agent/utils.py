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
    
    # Add eval critic data if double critic is enabled
    if hasattr(policy, 'use_double_critic') and policy.use_double_critic:
        episode_data['value_r_eval'] = []
        episode_data['value_c_eval'] = []
    
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
        episode_value_r_eval = []
        episode_value_c_eval = []
        
        for _ in range(max_ep_len):
            with torch.no_grad():
                risk = torch.exp(risk_model(obs)) if use_risk else None
                _, _, value_r, value_c = policy.step(obs, risk, deterministic=True)
                
                # Get eval critic values if double critic is enabled
                if hasattr(policy, 'use_double_critic') and policy.use_double_critic:
                    value_r_eval, value_c_eval = policy.get_eval_values(obs, risk)
                    episode_value_r_eval.append(value_r_eval)
                    episode_value_c_eval.append(value_c_eval)
            
            episode_obs.append(obs)
            episode_value_r.append(value_r)
            episode_value_c.append(value_c)
            
            action = policy.actor(obs, risk).sample()
            # Vector envs expect numpy actions; avoid numpy * Tensor type errors.
            action_np = action.detach().cpu().numpy()

            if "Safe" in args.task:
                next_obs, reward, cost, terminated, truncated, info = env.step(action_np)
                success = 0
            else:
                next_obs, reward, terminated, truncated, info = env.step(action_np)
                try:
                    cost = info["cost"]
                    success = info["success"]
                except:
                    cost = terminated
                    success = 0 
            
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
                    if hasattr(policy, 'use_double_critic') and policy.use_double_critic:
                        episode_data['value_r_eval'].append(torch.stack([vr[env_idx] for vr in episode_value_r_eval]))
                        episode_data['value_c_eval'].append(torch.stack([vc[env_idx] for vc in episode_value_c_eval]))
                    episodes_completed += 1
            
            if episodes_completed >= num_episodes:
                break
            
            if (terminated | truncated).all():
                break
                
        if episodes_completed >= num_episodes:
            break
            
    # print(torch.sum(torch.stack(episode_data["costs"])))
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
    color_values: Optional[torch.Tensor] = None,
    color_label: str = ""
) -> plt.Figure:
    """
    Create a scatter plot comparing value estimates and MC returns.
    
    Args:
        value_estimates: Tensor of predicted values
        monte_carlo_returns: Tensor of Monte Carlo returns
        title: Plot title
        
    Returns:
        matplotlib Figure object for wandb logging
    """
    # Convert to numpy
    values_np = value_estimates.detach().cpu().numpy().flatten()
    returns_np = monte_carlo_returns.detach().cpu().numpy().flatten()
    color_np = (
        color_values.detach().cpu().numpy().flatten()
        if color_values is not None
        else None
    )
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    
    # Create scatter plot with optional color coding
    if color_np is not None:
        scatter = plt.scatter(
            returns_np, values_np, c=color_np, alpha=0.25, cmap="turbo", s=10
        )
        plt.colorbar(scatter, label=color_label or "Value Discrepancy (Main - Eval)")
    else:
        plt.scatter(returns_np, values_np, alpha=0.2)
    
    # Calculate correlations
    corr = calculate_correlation(value_estimates, monte_carlo_returns)
    
    # Add correlation info to plot
    plt.title(f'{title}\nPearson: {corr["pearson_corr"]:.3f}, Spearman: {corr["spearman_corr"]:.3f}')
    plt.xlabel('Monte Carlo Returns')
    plt.ylabel('Predicted Values')
    # plt.legend()
    
    return fig

def create_critic_comparison_plot(
    main_critic_values: torch.Tensor,
    eval_critic_values: torch.Tensor,
    title: str,
    value_type: str = "Reward"
) -> plt.Figure:
    """
    Create a scatter plot comparing predictions from main critic vs eval critic.
    
    Args:
        main_critic_values: Tensor of values from main critic (used for policy updates)
        eval_critic_values: Tensor of values from eval critic (not used for policy updates)
        title: Plot title
        value_type: Type of value being compared ("Reward" or "Cost")
        
    Returns:
        matplotlib Figure object for wandb logging
    """
    # Convert to numpy
    main_np = main_critic_values.detach().cpu().numpy().flatten()
    eval_np = eval_critic_values.detach().cpu().numpy().flatten()
    
    # Calculate discrepancy
    discrepancy = main_np - eval_np
    mean_discrepancy = np.mean(discrepancy)
    std_discrepancy = np.std(discrepancy)
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    
    # Create scatter plot with color coding by discrepancy
    scatter = plt.scatter(main_np, eval_np, c=discrepancy, alpha=0.4, cmap='seismic', s=10)
    plt.colorbar(scatter, label='Discrepancy (Main - Eval)')
    
    # Add diagonal line (y=x) for perfect agreement
    min_val = min(main_np.min(), eval_np.min())
    max_val = max(main_np.max(), eval_np.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Agreement (y=x)')
    
    # Calculate correlation
    corr = calculate_correlation(main_critic_values, eval_critic_values)
    
    # Add title and labels
    plt.title(f'{title}\nMean Discrepancy: {mean_discrepancy:.4f} ± {std_discrepancy:.4f}\n'
              f'Pearson: {corr["pearson_corr"]:.3f}, Spearman: {corr["spearman_corr"]:.3f}')
    plt.xlabel(f'Main Critic {value_type} Value (used for policy updates)', fontsize=11)
    plt.ylabel(f'Eval Critic {value_type} Value (evaluation only)', fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text annotation about interpretation
    if mean_discrepancy > 0:
        bias_text = f'Main critic overestimates by {mean_discrepancy:.4f} on average'
    else:
        bias_text = f'Main critic underestimates by {abs(mean_discrepancy):.4f} on average'
    
    plt.text(0.05, 0.95, bias_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    return fig

def evaluate_value_estimation_error(
    value_estimates: torch.Tensor,
    monte_carlo_returns: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    create_plot: bool = False,
    plot_title: str = "",
    color_values: Optional[torch.Tensor] = None,
    color_label: str = ""
) -> Dict[str, float]:
    """
    Calculate metrics to evaluate value function estimation bias.
    
    Args:
        value_estimates: Tensor of predicted values
        monte_carlo_returns: Tensor of Monte Carlo returns
        mask: Optional tensor to mask certain timesteps
        create_plot: Whether to create scatter plot
        plot_title: Title for the scatter plot
        
    Returns:
        Dictionary containing evaluation metrics and optionally the plot figure
    """
    if mask is not None:
        value_estimates = value_estimates[mask]
        monte_carlo_returns = monte_carlo_returns[mask]
        if color_values is not None:
            color_values = color_values[mask]
    
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
            color_values=color_values,
            color_label=color_label
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
        'error': errors,
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
        Dictionary containing evaluation metrics for both reward and cost critics,
        and discrepancy metrics if double critic is enabled
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

    # Optional discrepancy for color-coding if double critic is enabled
    reward_color_values = None
    cost_color_values = None
    if hasattr(policy, 'use_double_critic') and policy.use_double_critic and 'value_r_eval' in episode_data:
        all_value_r_eval = torch.cat(episode_data['value_r_eval'])
        all_value_c_eval = torch.cat(episode_data['value_c_eval'])
        reward_color_values = torch.abs(all_value_r - all_value_r_eval)
        cost_color_values = torch.abs(all_value_c - all_value_c_eval)
    




    # Evaluate reward critic
    reward_metrics = evaluate_value_estimation_error(
        all_value_r,
        all_reward_returns,
        create_plot=create_plots,
        plot_title="Reward Value Estimates vs MC Returns",
        color_values=reward_color_values,
        color_label="Reward Discrepancy |Main - Eval|"
    )
    
    # Evaluate cost critic
    cost_metrics = evaluate_value_estimation_error(
        all_value_c,
        all_cost_returns,
        create_plot=create_plots,
        plot_title="Cost Value Estimates vs MC Returns",
        color_values=cost_color_values,
        color_label="Cost Discrepancy |Main - Eval|"
    )
    
    # Calculate correlation between std and estimation error
    reward_std_error_corr = calculate_correlation(reward_color_values, reward_metrics['error'])
    cost_std_error_corr = calculate_correlation(cost_color_values, cost_metrics['error'])

    reward_metrics['std_error_corr'] = reward_std_error_corr
    cost_metrics['std_error_corr'] = cost_std_error_corr
    result = {
        'reward_critic': reward_metrics,
        'cost_critic': cost_metrics
    }
    
    # Compute discrepancies between main and eval critics if double critic is enabled
    if hasattr(policy, 'use_double_critic') and policy.use_double_critic and 'value_r_eval' in episode_data:
        
        # Discrepancy = main_critic - eval_critic
        # Positive means main critic overestimates relative to eval critic
        reward_discrepancy = all_value_r - all_value_r_eval
        cost_discrepancy = all_value_c - all_value_c_eval
        
        reward_discrepancy_metrics = {
            'mean_discrepancy': reward_discrepancy.mean().item(),
            'std_discrepancy': reward_discrepancy.std().item(),
            'mean_abs_discrepancy': torch.abs(reward_discrepancy).mean().item(),
            'max_discrepancy': reward_discrepancy.max().item(),
            'min_discrepancy': reward_discrepancy.min().item(),
            'overestimate_ratio': (reward_discrepancy > 0).float().mean().item(),
            'underestimate_ratio': (reward_discrepancy < 0).float().mean().item(),
            'mean_main_value': all_value_r.mean().item(),
            'mean_eval_value': all_value_r_eval.mean().item(),
        }
        
        cost_discrepancy_metrics = {
            'mean_discrepancy': cost_discrepancy.mean().item(),
            'std_discrepancy': cost_discrepancy.std().item(),
            'mean_abs_discrepancy': torch.abs(cost_discrepancy).mean().item(),
            'max_discrepancy': cost_discrepancy.max().item(),
            'min_discrepancy': cost_discrepancy.min().item(),
            'overestimate_ratio': (cost_discrepancy > 0).float().mean().item(),
            'underestimate_ratio': (cost_discrepancy < 0).float().mean().item(),
            'mean_main_value': all_value_c.mean().item(),
            'mean_eval_value': all_value_c_eval.mean().item(),
        }
        
        # Create comparison scatter plots
        if create_plots:
            reward_comparison_fig = create_critic_comparison_plot(
                all_value_r,
                all_value_r_eval,
                title=f"Reward Critic Comparison: Main vs Eval",
                value_type="Reward"
            )
            reward_discrepancy_metrics['comparison_plot'] = reward_comparison_fig
            
            cost_comparison_fig = create_critic_comparison_plot(
                all_value_c,
                all_value_c_eval,
                title=f"Cost Critic Comparison: Main vs Eval",
                value_type="Cost"
            )
            cost_discrepancy_metrics['comparison_plot'] = cost_comparison_fig
        
        result['reward_discrepancy'] = reward_discrepancy_metrics
        result['cost_discrepancy'] = cost_discrepancy_metrics
    
    return result

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
    gamma: float = 0.99
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
    
    # Evaluate reward critic
    reward_metrics = evaluate_value_estimation_error(
        buffer_data['value_r'],
        reward_returns
    )
    
    # Evaluate cost critic
    cost_metrics = evaluate_value_estimation_error(
        buffer_data['value_c'],
        cost_returns
    )
    
    return {
        'reward_critic': reward_metrics,
        'cost_critic': cost_metrics
    } 