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

"""
Utility functions for SAM (Sharpness-Aware Minimization) analysis and visualization.
Contains plotting functions for likelihood ratio analysis and SAM vs Standard TRPO comparisons.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy import stats

matplotlib.use('Agg')  # Use non-interactive backend


def create_likelihood_ratio_scatter_plots(pre_ratios, post_ratios, reward_adv, cost_adv, lagrangian_adv, epoch, use_transparency=True):
    """
    Create scatter plots of likelihood ratios vs different types of advantages.
    
    Args:
        pre_ratios: Pre-update likelihood ratios
        post_ratios: Post-update likelihood ratios  
        reward_adv: Reward advantages
        cost_adv: Cost advantages
        lagrangian_adv: Lagrangian advantages (combined)
        epoch: Current epoch number
        use_transparency: Whether to use transparency based on pre-update likelihood ratios
    
    Returns:
        dict: Dictionary containing matplotlib figures for each plot
    """
    # Convert tensors to numpy for plotting
    pre_ratios_np = pre_ratios.detach().cpu().numpy()
    post_ratios_np = post_ratios.detach().cpu().numpy()
    reward_adv_np = reward_adv.detach().cpu().numpy()
    cost_adv_np = cost_adv.detach().cpu().numpy()
    lagrangian_adv_np = lagrangian_adv.detach().cpu().numpy()
    
    # Compute transparency based on pre-update likelihood ratios
    if use_transparency:
        # Normalize pre-update ratios to [0, 1] range for transparency
        # Low likelihood (close to 0) -> high transparency (dark)
        # High likelihood (close to 1) -> low transparency (light)
        pre_ratios_normalized = (pre_ratios_np - pre_ratios_np.min()) / (pre_ratios_np.max() - pre_ratios_np.min() + 1e-8)
        # Invert so low likelihood = high alpha (dark), high likelihood = low alpha (light)
        alpha_values = 1.0 - pre_ratios_normalized
        # Scale to reasonable alpha range [0.3, 0.9]
        alpha_values = 0.3 + 0.6 * alpha_values
    else:
        # Default alpha if transparency not used
        alpha_values = 0.6
    
    plots = {}
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Likelihood Ratios vs Advantages (Epoch {epoch})', fontsize=16)
    
    # Pre-update ratios vs advantages
    axes[0, 0].scatter(reward_adv_np, pre_ratios_np, s=10, alpha=alpha_values)
    axes[0, 0].set_xlabel('Reward Advantage')
    axes[0, 0].set_ylabel('Pre-Update Likelihood Ratio')
    axes[0, 0].set_title('Pre-Update Ratios vs Reward Advantage')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].scatter(cost_adv_np, pre_ratios_np, s=10, alpha=alpha_values)
    axes[0, 1].set_xlabel('Cost Advantage')
    axes[0, 1].set_ylabel('Pre-Update Likelihood Ratio')
    axes[0, 1].set_title('Pre-Update Ratios vs Cost Advantage')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].scatter(lagrangian_adv_np, pre_ratios_np, s=10, alpha=alpha_values)
    axes[0, 2].set_xlabel('Lagrangian Advantage')
    axes[0, 2].set_ylabel('Pre-Update Likelihood Ratio')
    axes[0, 2].set_title('Pre-Update Ratios vs Lagrangian Advantage')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Post-update ratios vs advantages
    axes[1, 0].scatter(reward_adv_np, post_ratios_np, s=10, color='red', alpha=alpha_values)
    axes[1, 0].set_xlabel('Reward Advantage')
    axes[1, 0].set_ylabel('Post-Update Likelihood Ratio')
    axes[1, 0].set_title('Post-Update Ratios vs Reward Advantage')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].scatter(cost_adv_np, post_ratios_np, s=10, color='red', alpha=alpha_values)
    axes[1, 1].set_xlabel('Cost Advantage')
    axes[1, 1].set_ylabel('Post-Update Likelihood Ratio')
    axes[1, 1].set_title('Post-Update Ratios vs Cost Advantage')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].scatter(lagrangian_adv_np, post_ratios_np, s=10, color='red', alpha=alpha_values)
    axes[1, 2].set_xlabel('Lagrangian Advantage')
    axes[1, 2].set_ylabel('Post-Update Likelihood Ratio')
    axes[1, 2].set_title('Post-Update Ratios vs Lagrangian Advantage')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plots['combined'] = fig
    
    # Create individual plots for better visibility
    # Pre-update vs Reward Advantage
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    ax1.scatter(reward_adv_np, pre_ratios_np, s=15, alpha=alpha_values)
    ax1.set_xlabel('Reward Advantage')
    ax1.set_ylabel('Pre-Update Likelihood Ratio')
    ax1.set_title(f'Pre-Update Likelihood Ratios vs Reward Advantage (Epoch {epoch})')
    ax1.grid(True, alpha=0.3)
    plots['pre_vs_reward'] = fig1
    
    # Post-update vs Reward Advantage
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    ax2.scatter(reward_adv_np, post_ratios_np, s=15, color='red', alpha=alpha_values)
    ax2.set_xlabel('Reward Advantage')
    ax2.set_ylabel('Post-Update Likelihood Ratio')
    ax2.set_title(f'Post-Update Likelihood Ratios vs Reward Advantage (Epoch {epoch})')
    ax2.grid(True, alpha=0.3)
    plots['post_vs_reward'] = fig2
    
    # Pre-update vs Cost Advantage
    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 6))
    ax3.scatter(cost_adv_np, pre_ratios_np, s=15, alpha=alpha_values)
    ax3.set_xlabel('Cost Advantage')
    ax3.set_ylabel('Pre-Update Likelihood Ratio')
    ax3.set_title(f'Pre-Update Likelihood Ratios vs Cost Advantage (Epoch {epoch})')
    ax3.grid(True, alpha=0.3)
    plots['pre_vs_cost'] = fig3
    
    # Post-update vs Cost Advantage
    fig4, ax4 = plt.subplots(1, 1, figsize=(8, 6))
    ax4.scatter(cost_adv_np, post_ratios_np, s=15, color='red', alpha=alpha_values)
    ax4.set_xlabel('Cost Advantage')
    ax4.set_ylabel('Post-Update Likelihood Ratio')
    ax4.set_title(f'Post-Update Likelihood Ratios vs Cost Advantage (Epoch {epoch})')
    ax4.grid(True, alpha=0.3)
    plots['post_vs_cost'] = fig4
    
    # Pre-update vs Lagrangian Advantage
    fig5, ax5 = plt.subplots(1, 1, figsize=(8, 6))
    ax5.scatter(lagrangian_adv_np, pre_ratios_np, s=15, alpha=alpha_values)
    ax5.set_xlabel('Lagrangian Advantage')
    ax5.set_ylabel('Pre-Update Likelihood Ratio')
    ax5.set_title(f'Pre-Update Likelihood Ratios vs Lagrangian Advantage (Epoch {epoch})')
    ax5.grid(True, alpha=0.3)
    plots['pre_vs_lagrangian'] = fig5
    
    # Post-update vs Lagrangian Advantage
    fig6, ax6 = plt.subplots(1, 1, figsize=(8, 6))
    ax6.scatter(lagrangian_adv_np, post_ratios_np, s=15, color='red', alpha=alpha_values)
    ax6.set_xlabel('Lagrangian Advantage')
    ax6.set_ylabel('Post-Update Likelihood Ratio')
    ax6.set_title(f'Post-Update Likelihood Ratios vs Lagrangian Advantage (Epoch {epoch})')
    ax6.grid(True, alpha=0.3)
    plots['post_vs_lagrangian'] = fig6
    
    return plots


def create_sam_vs_standard_comparison_plots(pre_ratios, sam_post_ratios, standard_post_ratios, 
                                           reward_adv, cost_adv, lagrangian_adv, epoch):
    """
    Create comparison plots between SAM and standard TRPO updates.
    
    Args:
        pre_ratios: Pre-update likelihood ratios
        sam_post_ratios: Post-update likelihood ratios using SAM
        standard_post_ratios: Post-update likelihood ratios using standard TRPO
        reward_adv: Reward advantages
        cost_adv: Cost advantages
        lagrangian_adv: Lagrangian advantages (combined)
        epoch: Current epoch number
    
    Returns:
        dict: Dictionary containing matplotlib figures for comparison plots
    """
    # Convert tensors to numpy for plotting
    pre_ratios_np = pre_ratios.detach().cpu().numpy()
    sam_post_ratios_np = sam_post_ratios.detach().cpu().numpy()
    standard_post_ratios_np = standard_post_ratios.detach().cpu().numpy()
    reward_adv_np = reward_adv.detach().cpu().numpy()
    cost_adv_np = cost_adv.detach().cpu().numpy()
    lagrangian_adv_np = lagrangian_adv.detach().cpu().numpy()
    
    plots = {}
    
    # Create comparison figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'SAM vs Standard TRPO Likelihood Ratios (Epoch {epoch})', fontsize=16)
    
    # SAM vs Standard comparison for each advantage type
    # Reward Advantage
    axes[0, 0].scatter(reward_adv_np, sam_post_ratios_np, alpha=0.6, s=10, color='red', label='SAM')
    axes[0, 0].scatter(reward_adv_np, standard_post_ratios_np, alpha=0.6, s=10, color='blue', label='Standard')
    axes[0, 0].set_xlabel('Reward Advantage')
    axes[0, 0].set_ylabel('Post-Update Likelihood Ratio')
    axes[0, 0].set_title('SAM vs Standard: Reward Advantage')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Cost Advantage
    axes[0, 1].scatter(cost_adv_np, sam_post_ratios_np, alpha=0.6, s=10, color='red', label='SAM')
    axes[0, 1].scatter(cost_adv_np, standard_post_ratios_np, alpha=0.6, s=10, color='blue', label='Standard')
    axes[0, 1].set_xlabel('Cost Advantage')
    axes[0, 1].set_ylabel('Post-Update Likelihood Ratio')
    axes[0, 1].set_title('SAM vs Standard: Cost Advantage')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Lagrangian Advantage
    axes[0, 2].scatter(lagrangian_adv_np, sam_post_ratios_np, alpha=0.6, s=10, color='red', label='SAM')
    axes[0, 2].scatter(lagrangian_adv_np, standard_post_ratios_np, alpha=0.6, s=10, color='blue', label='Standard')
    axes[0, 2].set_xlabel('Lagrangian Advantage')
    axes[0, 2].set_ylabel('Post-Update Likelihood Ratio')
    axes[0, 2].set_title('SAM vs Standard: Lagrangian Advantage')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Ratio difference plots (SAM - Standard) vs advantages
    sam_minus_std = sam_post_ratios_np - standard_post_ratios_np
    
    axes[1, 0].scatter(reward_adv_np, sam_minus_std, alpha=0.6, s=10, color='green')
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Reward Advantage')
    axes[1, 0].set_ylabel('SAM - Standard Ratio Difference')
    axes[1, 0].set_title('Ratio Difference vs Reward Advantage')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].scatter(cost_adv_np, sam_minus_std, alpha=0.6, s=10, color='green')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Cost Advantage')
    axes[1, 1].set_ylabel('SAM - Standard Ratio Difference')
    axes[1, 1].set_title('Ratio Difference vs Cost Advantage')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].scatter(lagrangian_adv_np, sam_minus_std, alpha=0.6, s=10, color='green')
    axes[1, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 2].set_xlabel('Lagrangian Advantage')
    axes[1, 2].set_ylabel('SAM - Standard Ratio Difference')
    axes[1, 2].set_title('Ratio Difference vs Lagrangian Advantage')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plots['sam_vs_standard_combined'] = fig
    
    # Create individual comparison plots
    # SAM vs Standard - Reward
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    ax1.scatter(reward_adv_np, sam_post_ratios_np, alpha=0.6, s=15, color='red', label='SAM')
    ax1.scatter(reward_adv_np, standard_post_ratios_np, alpha=0.6, s=15, color='blue', label='Standard')
    ax1.set_xlabel('Reward Advantage')
    ax1.set_ylabel('Post-Update Likelihood Ratio')
    ax1.set_title(f'SAM vs Standard TRPO: Reward Advantage (Epoch {epoch})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plots['sam_vs_standard_reward'] = fig1
    
    # SAM vs Standard - Cost
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    ax2.scatter(cost_adv_np, sam_post_ratios_np, alpha=0.6, s=15, color='red', label='SAM')
    ax2.scatter(cost_adv_np, standard_post_ratios_np, alpha=0.6, s=15, color='blue', label='Standard')
    ax2.set_xlabel('Cost Advantage')
    ax2.set_ylabel('Post-Update Likelihood Ratio')
    ax2.set_title(f'SAM vs Standard TRPO: Cost Advantage (Epoch {epoch})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plots['sam_vs_standard_cost'] = fig2
    
    # SAM vs Standard - Lagrangian
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
    ax3.scatter(lagrangian_adv_np, sam_post_ratios_np, alpha=0.6, s=15, color='red', label='SAM')
    ax3.scatter(lagrangian_adv_np, standard_post_ratios_np, alpha=0.6, s=15, color='blue', label='Standard')
    ax3.set_xlabel('Lagrangian Advantage')
    ax3.set_ylabel('Post-Update Likelihood Ratio')
    ax3.set_title(f'SAM vs Standard TRPO: Lagrangian Advantage (Epoch {epoch})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plots['sam_vs_standard_lagrangian'] = fig3
    
    # Ratio difference plots
    fig4, ax4 = plt.subplots(1, 1, figsize=(10, 6))
    ax4.scatter(reward_adv_np, sam_minus_std, alpha=0.6, s=15, color='green')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Reward Advantage')
    ax4.set_ylabel('SAM - Standard Ratio Difference')
    ax4.set_title(f'Likelihood Ratio Difference vs Reward Advantage (Epoch {epoch})')
    ax4.grid(True, alpha=0.3)
    plots['ratio_difference_reward'] = fig4
    
    fig5, ax5 = plt.subplots(1, 1, figsize=(10, 6))
    ax5.scatter(cost_adv_np, sam_minus_std, alpha=0.6, s=15, color='green')
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Cost Advantage')
    ax5.set_ylabel('SAM - Standard Ratio Difference')
    ax5.set_title(f'Likelihood Ratio Difference vs Cost Advantage (Epoch {epoch})')
    ax5.grid(True, alpha=0.3)
    plots['ratio_difference_cost'] = fig5
    
    fig6, ax6 = plt.subplots(1, 1, figsize=(10, 6))
    ax6.scatter(lagrangian_adv_np, sam_minus_std, alpha=0.6, s=15, color='green')
    ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Lagrangian Advantage')
    ax6.set_ylabel('SAM - Standard Ratio Difference')
    ax6.set_title(f'Likelihood Ratio Difference vs Lagrangian Advantage (Epoch {epoch})')
    ax6.grid(True, alpha=0.3)
    plots['ratio_difference_lagrangian'] = fig6
    
    # Policy difference plots - show how the actual policy changes differ
    # Compute policy differences (log probability differences)
    sam_policy_diff = sam_post_ratios_np - pre_ratios_np  # SAM policy change
    standard_policy_diff = standard_post_ratios_np - pre_ratios_np  # Standard policy change
    policy_diff_diff = sam_policy_diff - standard_policy_diff  # Difference in policy changes
    
    # Policy change comparison plots
    fig7, ax7 = plt.subplots(1, 1, figsize=(10, 6))
    ax7.scatter(reward_adv_np, sam_policy_diff, alpha=0.6, s=15, color='red', label='SAM Policy Change')
    ax7.scatter(reward_adv_np, standard_policy_diff, alpha=0.6, s=15, color='blue', label='Standard Policy Change')
    ax7.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax7.set_xlabel('Reward Advantage')
    ax7.set_ylabel('Policy Change (Post - Pre Ratio)')
    ax7.set_title(f'Policy Change Comparison vs Reward Advantage (Epoch {epoch})')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    plots['policy_change_reward'] = fig7
    
    fig8, ax8 = plt.subplots(1, 1, figsize=(10, 6))
    ax8.scatter(cost_adv_np, sam_policy_diff, alpha=0.6, s=15, color='red', label='SAM Policy Change')
    ax8.scatter(cost_adv_np, standard_policy_diff, alpha=0.6, s=15, color='blue', label='Standard Policy Change')
    ax8.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax8.set_xlabel('Cost Advantage')
    ax8.set_ylabel('Policy Change (Post - Pre Ratio)')
    ax8.set_title(f'Policy Change Comparison vs Cost Advantage (Epoch {epoch})')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    plots['policy_change_cost'] = fig8
    
    fig9, ax9 = plt.subplots(1, 1, figsize=(10, 6))
    ax9.scatter(lagrangian_adv_np, sam_policy_diff, alpha=0.6, s=15, color='red', label='SAM Policy Change')
    ax9.scatter(lagrangian_adv_np, standard_policy_diff, alpha=0.6, s=15, color='blue', label='Standard Policy Change')
    ax9.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax9.set_xlabel('Lagrangian Advantage')
    ax9.set_ylabel('Policy Change (Post - Pre Ratio)')
    ax9.set_title(f'Policy Change Comparison vs Lagrangian Advantage (Epoch {epoch})')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    plots['policy_change_lagrangian'] = fig9
    
    # Policy change difference plots
    fig10, ax10 = plt.subplots(1, 1, figsize=(10, 6))
    ax10.scatter(reward_adv_np, policy_diff_diff, alpha=0.6, s=15, color='purple')
    ax10.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax10.set_xlabel('Reward Advantage')
    ax10.set_ylabel('SAM - Standard Policy Change Difference')
    ax10.set_title(f'Policy Change Difference vs Reward Advantage (Epoch {epoch})')
    ax10.grid(True, alpha=0.3)
    plots['policy_change_diff_reward'] = fig10
    
    fig11, ax11 = plt.subplots(1, 1, figsize=(10, 6))
    ax11.scatter(cost_adv_np, policy_diff_diff, alpha=0.6, s=15, color='purple')
    ax11.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax11.set_xlabel('Cost Advantage')
    ax11.set_ylabel('SAM - Standard Policy Change Difference')
    ax11.set_title(f'Policy Change Difference vs Cost Advantage (Epoch {epoch})')
    ax11.grid(True, alpha=0.3)
    plots['policy_change_diff_cost'] = fig11
    
    fig12, ax12 = plt.subplots(1, 1, figsize=(10, 6))
    ax12.scatter(lagrangian_adv_np, policy_diff_diff, alpha=0.6, s=15, color='purple')
    ax12.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax12.set_xlabel('Lagrangian Advantage')
    ax12.set_ylabel('SAM - Standard Policy Change Difference')
    ax12.set_title(f'Policy Change Difference vs Lagrangian Advantage (Epoch {epoch})')
    ax12.grid(True, alpha=0.3)
    plots['policy_change_diff_lagrangian'] = fig12
    
    return plots


def create_likelihood_ratio_difference_correlation_plots(sam_post_ratios, standard_post_ratios, 
                                                       reward_adv, cost_adv, lagrangian_adv, epoch, pre_ratios=None):
    """
    Create scatter plots showing likelihood ratio differences with correlation lines.
    
    Args:
        sam_post_ratios: Post-update likelihood ratios using SAM
        standard_post_ratios: Post-update likelihood ratios using standard TRPO
        reward_adv: Reward advantages
        cost_adv: Cost advantages
        lagrangian_adv: Lagrangian advantages (combined)
        epoch: Current epoch number
        pre_ratios: Pre-update likelihood ratios for transparency mapping
    
    Returns:
        dict: Dictionary containing matplotlib figures for correlation plots
    """
    # Convert tensors to numpy for plotting
    sam_post_ratios_np = sam_post_ratios.detach().cpu().numpy()
    standard_post_ratios_np = standard_post_ratios.detach().cpu().numpy()
    reward_adv_np = reward_adv.detach().cpu().numpy()
    cost_adv_np = cost_adv.detach().cpu().numpy()
    lagrangian_adv_np = lagrangian_adv.detach().cpu().numpy()
    
    # Compute likelihood ratio differences
    ratio_diff = sam_post_ratios_np - standard_post_ratios_np
    
    # Compute transparency based on pre-update likelihood ratios
    if pre_ratios is not None:
        pre_ratios_np = pre_ratios.detach().cpu().numpy()
        # Normalize pre-update ratios to [0, 1] range for transparency
        # Low likelihood (close to 0) -> high transparency (dark)
        # High likelihood (close to 1) -> low transparency (light)
        pre_ratios_normalized = (pre_ratios_np - pre_ratios_np.min()) / (pre_ratios_np.max() - pre_ratios_np.min() + 1e-8)
        # Invert so low likelihood = high alpha (dark), high likelihood = low alpha (light)
        alpha_values = 1.0 - pre_ratios_normalized
        # Scale to reasonable alpha range [0.3, 0.9]
        alpha_values = 0.1 + 0.8 * alpha_values
    else:
        # Default alpha if no pre_ratios provided
        alpha_values = 0.6
    
    plots = {}
    
    # Create individual correlation plots for each advantage type
    # Reward Advantage
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    scatter1 = ax1.scatter(reward_adv_np, ratio_diff, s=20, color='blue', 
                          edgecolors='black', linewidth=0.5, alpha=alpha_values)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # Add correlation line
    if len(reward_adv_np) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(reward_adv_np, ratio_diff)
        line = slope * reward_adv_np + intercept
        ax1.plot(reward_adv_np, line, 'r-', linewidth=2, alpha=0.8, 
                label=f'Correlation: r={r_value:.3f}, p={p_value:.3f}')
        ax1.legend()
    
    ax1.set_xlabel('Reward Advantage', fontsize=12)
    ax1.set_ylabel('Likelihood Ratio Difference (SAM - Standard)', fontsize=12)
    ax1.set_title(f'Likelihood Ratio Difference vs Reward Advantage (Epoch {epoch})', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=10)
    plots['ratio_diff_correlation_reward'] = fig1
    
    # Cost Advantage
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    scatter2 = ax2.scatter(cost_adv_np, ratio_diff, s=20, color='green', 
                          edgecolors='black', linewidth=0.5, alpha=alpha_values)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # Add correlation line
    if len(cost_adv_np) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(cost_adv_np, ratio_diff)
        line = slope * cost_adv_np + intercept
        ax2.plot(cost_adv_np, line, 'r-', linewidth=2, alpha=0.8,
                label=f'Correlation: r={r_value:.3f}, p={p_value:.3f}')
        ax2.legend()
    
    ax2.set_xlabel('Cost Advantage', fontsize=12)
    ax2.set_ylabel('Likelihood Ratio Difference (SAM - Standard)', fontsize=12)
    ax2.set_title(f'Likelihood Ratio Difference vs Cost Advantage (Epoch {epoch})', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=10)
    plots['ratio_diff_correlation_cost'] = fig2
    
    # Lagrangian Advantage
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 8))
    scatter3 = ax3.scatter(lagrangian_adv_np, ratio_diff, s=20, color='purple', 
                          edgecolors='black', linewidth=0.5, alpha=alpha_values)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # Add correlation line
    if len(lagrangian_adv_np) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(lagrangian_adv_np, ratio_diff)
        line = slope * lagrangian_adv_np + intercept
        ax3.plot(lagrangian_adv_np, line, 'r-', linewidth=2, alpha=0.8,
                label=f'Correlation: r={r_value:.3f}, p={p_value:.3f}')
        ax3.legend()
    
    ax3.set_xlabel('Lagrangian Advantage', fontsize=12)
    ax3.set_ylabel('Likelihood Ratio Difference (SAM - Standard)', fontsize=12)
    ax3.set_title(f'Likelihood Ratio Difference vs Lagrangian Advantage (Epoch {epoch})', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=10)
    plots['ratio_diff_correlation_lagrangian'] = fig3
    
    # Create a combined plot with all three advantage types
    fig4, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig4.suptitle(f'Likelihood Ratio Difference Correlations (Epoch {epoch})', fontsize=16)
    
    # Reward Advantage
    axes[0].scatter(reward_adv_np, ratio_diff, s=15, color='blue', 
                   edgecolors='black', linewidth=0.3, alpha=alpha_values)
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    if len(reward_adv_np) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(reward_adv_np, ratio_diff)
        line = slope * reward_adv_np + intercept
        axes[0].plot(reward_adv_np, line, 'r-', linewidth=2, alpha=0.8)
        axes[0].text(0.05, 0.95, f'r={r_value:.3f}\np={p_value:.3f}', 
                    transform=axes[0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[0].set_xlabel('Reward Advantage')
    axes[0].set_ylabel('Ratio Difference')
    axes[0].set_title('vs Reward Advantage')
    axes[0].grid(True, alpha=0.3)
    
    # Cost Advantage
    axes[1].scatter(cost_adv_np, ratio_diff, s=15, color='green', 
                   edgecolors='black', linewidth=0.3, alpha=alpha_values)
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    if len(cost_adv_np) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(cost_adv_np, ratio_diff)
        line = slope * cost_adv_np + intercept
        axes[1].plot(cost_adv_np, line, 'r-', linewidth=2, alpha=0.8)
        axes[1].text(0.05, 0.95, f'r={r_value:.3f}\np={p_value:.3f}', 
                    transform=axes[1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[1].set_xlabel('Cost Advantage')
    axes[1].set_ylabel('Ratio Difference')
    axes[1].set_title('vs Cost Advantage')
    axes[1].grid(True, alpha=0.3)
    
    # Lagrangian Advantage
    axes[2].scatter(lagrangian_adv_np, ratio_diff, s=15, color='purple', 
                   edgecolors='black', linewidth=0.3, alpha=alpha_values)
    axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    if len(lagrangian_adv_np) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(lagrangian_adv_np, ratio_diff)
        line = slope * lagrangian_adv_np + intercept
        axes[2].plot(lagrangian_adv_np, line, 'r-', linewidth=2, alpha=0.8)
        axes[2].text(0.05, 0.95, f'r={r_value:.3f}\np={p_value:.3f}', 
                    transform=axes[2].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[2].set_xlabel('Lagrangian Advantage')
    axes[2].set_ylabel('Ratio Difference')
    axes[2].set_title('vs Lagrangian Advantage')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plots['ratio_diff_correlation_combined'] = fig4
    
    return plots


def compute_correlation_metrics(sam_post_ratios, standard_post_ratios, reward_adv, cost_adv, lagrangian_adv):
    """
    Compute correlation metrics between likelihood ratio differences and advantages.
    
    Args:
        sam_post_ratios: Post-update likelihood ratios using SAM
        standard_post_ratios: Post-update likelihood ratios using standard TRPO
        reward_adv: Reward advantages
        cost_adv: Cost advantages
        lagrangian_adv: Lagrangian advantages (combined)
    
    Returns:
        dict: Dictionary containing correlation metrics
    """
    # Convert tensors to numpy for computation
    sam_post_ratios_np = sam_post_ratios.detach().cpu().numpy()
    standard_post_ratios_np = standard_post_ratios.detach().cpu().numpy()
    reward_adv_np = reward_adv.detach().cpu().numpy()
    cost_adv_np = cost_adv.detach().cpu().numpy()
    lagrangian_adv_np = lagrangian_adv.detach().cpu().numpy()
    
    # Compute likelihood ratio differences
    ratio_diff = sam_post_ratios_np - standard_post_ratios_np
    
    metrics = {}
    
    # Compute correlations for each advantage type
    if len(reward_adv_np) > 1:
        # Reward advantage correlations
        reward_corr, reward_p = stats.pearsonr(reward_adv_np, ratio_diff)
        reward_spearman_corr, reward_spearman_p = stats.spearmanr(reward_adv_np, ratio_diff)
        
        metrics.update({
            "SAM_Std_Correlation/RewardAdv_Pearson_r": reward_corr,
            "SAM_Std_Correlation/RewardAdv_Pearson_p": reward_p,
            "SAM_Std_Correlation/RewardAdv_Spearman_r": reward_spearman_corr,
            "SAM_Std_Correlation/RewardAdv_Spearman_p": reward_spearman_p,
            "SAM_Std_Correlation/RewardAdv_Significant": reward_p < 0.05,
        })
        
        # Cost advantage correlations
        cost_corr, cost_p = stats.pearsonr(cost_adv_np, ratio_diff)
        cost_spearman_corr, cost_spearman_p = stats.spearmanr(cost_adv_np, ratio_diff)
        
        metrics.update({
            "SAM_Std_Correlation/CostAdv_Pearson_r": cost_corr,
            "SAM_Std_Correlation/CostAdv_Pearson_p": cost_p,
            "SAM_Std_Correlation/CostAdv_Spearman_r": cost_spearman_corr,
            "SAM_Std_Correlation/CostAdv_Spearman_p": cost_spearman_p,
            "SAM_Std_Correlation/CostAdv_Significant": cost_p < 0.05,
        })
        
        # Lagrangian advantage correlations
        lagrangian_corr, lagrangian_p = stats.pearsonr(lagrangian_adv_np, ratio_diff)
        lagrangian_spearman_corr, lagrangian_spearman_p = stats.spearmanr(lagrangian_adv_np, ratio_diff)
        
        metrics.update({
            "SAM_Std_Correlation/LagrangianAdv_Pearson_r": lagrangian_corr,
            "SAM_Std_Correlation/LagrangianAdv_Pearson_p": lagrangian_p,
            "SAM_Std_Correlation/LagrangianAdv_Spearman_r": lagrangian_spearman_corr,
            "SAM_Std_Correlation/LagrangianAdv_Spearman_p": lagrangian_spearman_p,
            "SAM_Std_Correlation/LagrangianAdv_Significant": lagrangian_p < 0.05,
        })
        
        # Additional statistics
        metrics.update({
            "SAM_Std_Correlation/RatioDiff_Mean": ratio_diff.mean(),
            "SAM_Std_Correlation/RatioDiff_Std": ratio_diff.std(),
            "SAM_Std_Correlation/RatioDiff_Min": ratio_diff.min(),
            "SAM_Std_Correlation/RatioDiff_Max": ratio_diff.max(),
            "SAM_Std_Correlation/RatioDiff_AbsMean": np.abs(ratio_diff).mean(),
            "SAM_Std_Correlation/SAM_Larger_Count": np.sum(ratio_diff > 0),
            "SAM_Std_Correlation/Standard_Larger_Count": np.sum(ratio_diff < 0),
            "SAM_Std_Correlation/Equal_Count": np.sum(ratio_diff == 0),
            "SAM_Std_Correlation/SAM_Larger_Percentage": np.mean(ratio_diff > 0) * 100,
        })
        
        # Advantage statistics
        metrics.update({
            "SAM_Std_Correlation/RewardAdv_Mean": reward_adv_np.mean(),
            "SAM_Std_Correlation/RewardAdv_Std": reward_adv_np.std(),
            "SAM_Std_Correlation/CostAdv_Mean": cost_adv_np.mean(),
            "SAM_Std_Correlation/CostAdv_Std": cost_adv_np.std(),
            "SAM_Std_Correlation/LagrangianAdv_Mean": lagrangian_adv_np.mean(),
            "SAM_Std_Correlation/LagrangianAdv_Std": lagrangian_adv_np.std(),
        })
    
    return metrics
