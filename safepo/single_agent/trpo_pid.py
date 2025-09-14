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

import os
import random
import sys
import time
from collections import deque
from typing import Callable

import numpy as np
try: 
    from isaacgym import gymutil
except ImportError:
    pass
    
import wandb
import torch
import torch.nn as nn
import torch.optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from safepo.common.buffer import VectorizedOnPolicyBuffer
from safepo.common.env import make_sa_mujoco_env, make_sa_isaac_env, make_sa_safetygym_env, make_sa_gymrobot_env
from safepo.common.lagrange import PIDLagrangian as Lagrange
from safepo.common.logger import EpochLogger
from safepo.common.model import ActorVCritic
from safepo.utils.config import single_agent_args, isaac_gym_map, parse_sim_params
from safepo.single_agent.utils import *
from sam import *

CONJUGATE_GRADIENT_ITERS=15
TRPO_SEARCHING_STEPS=15

default_cfg = {
    'hidden_sizes': [64, 64],
    'gamma': 0.99,
    'target_kl': 0.01,
    'batch_size': 128,
    'learning_iters': 10,
    'max_grad_norm': 40.0,
    'use_layer_norm': False,
}

isaac_gym_specific_cfg = {
    'total_steps': 100000000,
    'steps_per_epoch': 32768,
    'hidden_sizes': [1024, 1024, 512],
    'gamma': 0.96,
    'target_kl': 0.016,
    'num_mini_batch': 4,
    'use_value_coefficient': True,
    'learning_iters': 8,
    'max_grad_norm': 1.0,
    'use_critic_norm': False,
    'use_layer_norm': False,
}



def render_and_save_gif(env, policy, device, max_steps=2000, use_risk=False, risk_model=None, gif_name="episode", wandb_log=True):
    """
    Renders an episode using the given policy and saves it as a gif.
    
    Args:
        env: The environment to render
        policy: The policy to use for actions
        device: The device to run the policy on
        max_steps: Maximum number of steps per episode
        use_risk: Whether to use risk model
        risk_model: Risk model to use if use_risk is True
        gif_name: Name to save the gif as
        wandb_log: Whether to log gif to wandb
    """
    import imageio
    import os
    
    frames = []
    obs, _ = env.reset()
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    
    done = False
    ep_len = 0
    
    while not done and ep_len < max_steps:
        # Render and save frame
        frame = env.render()
        frames.append(frame)
        
        # Get action from policy
        with torch.no_grad():
            risk = risk_model(obs) if use_risk else None
            action, _, _, _ = policy.step(obs, risk, deterministic=True)
        
        # Take step in environment
        action = action.detach().cpu().numpy()
        obs, _, terminated, truncated, _ = env.step(action)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        
        done = terminated or truncated
        ep_len += 1
    
    # Save gif
    gif_path = os.path.join(wandb.run.dir, f"{gif_name}.gif")
    imageio.mimsave(gif_path, frames, duration=30)
    
    # Log to wandb if requested
    if wandb_log:
        wandb.log({f"plots/{gif_name}": wandb.Image(gif_path)})
    
    # Clean up gif file if logged to wandb
    if wandb_log and os.path.exists(gif_path):
        os.remove(gif_path)
        
    return frames

# To ensure that the same observation normalization is used for both `env` and `eval_env`,
# you should share the same normalization statistics (e.g., running mean and variance)
# between them. If you are using a wrapper like NormalizeObservation or a custom
# normalization wrapper, you can do the following after creating both environments:

# Alternatively, if you use a wrapper class, you can pass the same instance or
# periodically sync the statistics:
def sync_obs_normalization(env, eval_env):
    if hasattr(env, 'obs_rms') and hasattr(eval_env, 'obs_rms'):
        eval_env.obs_rms.mean = env.obs_rms.mean.copy()
        eval_env.obs_rms.var = env.obs_rms.var.copy()
        eval_env.obs_rms.count = env.obs_rms.count

# Call this function before evaluation to sync stats
# sync_obs_normalization(env, eval_env)

# If you use a vectorized environment, ensure the normalization wrapper is applied
# identically to both, and share or sync the statistics as above.


def env_fn(env_id):
    if "Safety" in env_id:
        return make_sa_safetygym_env
    else:
        return make_sa_gymrobot_env




def main(args, cfg_env=None):
    # set the random seed, device and number of threads
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    device = torch.device(f'{args.device}:{args.device_id}')

    #run = wandb.init(config=vars(args), entity="manila95",
    #            project="risk_aware_exploration",
    #            monitor_gym=True,
    #            sync_tensorboard=True, save_code=True)

    risk_size = args.quantile_num if args.risk_type == "quantile" else 2
    risk_bins = np.array([i*args.quantile_size for i in range(args.quantile_num)])

    if args.task not in isaac_gym_map.keys():
        env, obs_space, act_space = env_fn(args.task)(
            args, num_envs=args.num_envs, env_id=args.task, seed=args.seed
        )
        eval_env, _, _ = env_fn(args.task)(args, num_envs=1, env_id=args.task, seed=None)
        config = default_cfg

    else:
        sim_params = parse_sim_params(cfg_env, None)
        env = make_sa_isaac_env(cfg=cfg_env, sim_params=sim_params)
        eval_env = env
        obs_space = env.observation_space
        act_space = env.action_space
        args.num_envs = env.num_envs
        config = isaac_gym_specific_cfg

    # set training steps
    steps_per_epoch = config.get("steps_per_epoch", args.steps_per_epoch)
    total_steps = config.get("total_steps", args.total_steps)
    local_steps_per_epoch = steps_per_epoch // args.num_envs
    epochs = total_steps // steps_per_epoch
    # create the actor-critic module
    policy = ActorVCritic(
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        hidden_sizes=config["hidden_sizes"],
        use_risk=args.use_risk,
        risk_size=risk_size,
        use_actor_layer_norm=args.use_actor_layer_norm,
        use_critic_layer_norm=args.use_critic_layer_norm,
    ).to(device)
    reward_critic_optimizer = torch.optim.Adam(
        policy.reward_critic.parameters(), lr=1e-3
    )
    cost_critic_optimizer = torch.optim.Adam(
        policy.cost_critic.parameters(), lr=1e-3
    )

    if args.use_risk:
        risk_model_class = {"bayesian": {"continuous": BayesRiskEstCont, "binary": BayesRiskEst, "quantile": BayesRiskEst}, 
                    "mlp": {"continuous": RiskEst, "binary": RiskEst}} 

        risk_model = BayesRiskEst(obs_size=obs_space.shape[0], batch_norm=True, out_size=risk_size)
        if os.path.exists(args.risk_model_path):
            risk_model.load_state_dict(torch.load(args.risk_model_path, map_location=device))

        risk_model.to(device)
        risk_model.eval()

        opt_risk = torch.optim.Adam(risk_model.parameters(), lr=args.risk_lr, eps=1e-10)

        if args.fine_tune_risk:
            rb = ReplayBuffer(args.total_steps, obs_space.shape[0], risk_size, device)

            if args.risk_type == "quantile":
                weight_tensor = torch.Tensor([1]*args.quantile_num).to(device)
                weight_tensor[0] = args.risk_weight
            elif args.risk_type == "binary":
                weight_tensor = torch.Tensor([1., args.risk_weight]).to(device)
            risk_criterion = nn.NLLLoss(weight=weight_tensor)

    # create the vectorized on-policy buffer
    buffer = VectorizedOnPolicyBuffer(
        obs_space=obs_space,
        act_space=act_space,
        size=local_steps_per_epoch,
        device=device,
        num_envs=args.num_envs,
        gamma=config["gamma"],
    )
    # setup lagrangian multiplier
    lagrange = Lagrange(
        cost_limit=args.cost_limit,
        lagrangian_multiplier_init=args.lagrangian_multiplier_init,
        pid_kd=args.pid_kd,
        pid_ki=args.pid_ki,
        pid_kp=args.pid_kp,
        penalty_max=args.lagrangian_multiplier_init*100,
    )
    # set up the logger
    dict_args = vars(args)
    dict_args.update(config)
    logger = EpochLogger(
        log_dir=args.log_dir,
        seed=str(args.seed),
    )
    rew_deque = deque(maxlen=50)
    cost_deque = deque(maxlen=50)
    len_deque = deque(maxlen=50)
    goal_deque = deque(maxlen=50)
    success_deque = deque(maxlen=50)
    eval_rew_deque = deque(maxlen=50)
    eval_cost_deque = deque(maxlen=50)
    eval_len_deque = deque(maxlen=50)
    eval_goal_deque = deque(maxlen=50)
    logger.save_config(dict_args)
    logger.setup_torch_saver(policy.actor)
    logger.log("Start with training.")
    obs, _ = env.reset()
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    ep_ret, ep_cost, ep_len, ep_goal, ep_success = (
        np.zeros(args.num_envs),
        np.zeros(args.num_envs),
        np.zeros(args.num_envs),
        np.zeros(args.num_envs),
        np.zeros(args.num_envs),
    )
    total_cost, eval_total_cost = 0, 0
    f_next_obs, f_costs = None, None

    global_step = 0
    total_violations = 0
    
    # training loop
    for epoch in range(epochs):
        rollout_start_time = time.time()
        # collect samples until we have enough to update
        for steps in range(local_steps_per_epoch):
            global_step += 1
            with torch.no_grad():
                    risk = risk_model(obs) if args.use_risk else None
                    act, log_prob, value_r, value_c = policy.step(obs, risk, deterministic=False)

            action = act.detach().squeeze() if args.task in isaac_gym_map.keys() else act.detach().squeeze().cpu().numpy()
            
            if "Safe" in args.task:
                next_obs, reward, cost, terminated, truncated, info = env.step(action)
                success = 0
            else:
                next_obs, reward, terminated, truncated, info = env.step(action)
                try:
                    cost = info["cost"]
                    success = info["success"]
                except:
                    cost = terminated
                    success = 0

            ep_ret += reward.cpu().numpy() if args.task in isaac_gym_map.keys() else reward
            ep_cost += cost.cpu().numpy() if args.task in isaac_gym_map.keys() else cost
            ep_success += success.cpu().numpy() if args.task in isaac_gym_map.keys() else success
            ep_len += 1
            next_obs, reward, cost, terminated, truncated = (
                torch.as_tensor(x, dtype=torch.float32, device=device)
                for x in (next_obs, reward, cost, terminated, truncated)
            )
            if args.use_risk and args.fine_tune_risk:
                f_next_obs = next_obs.unsqueeze(0) if f_next_obs is None else torch.concat([f_next_obs, next_obs.unsqueeze(0)], axis=0)
                f_costs = cost.unsqueeze(0) if f_costs is None else torch.concat([f_costs, cost.unsqueeze(0)], axis=0)
            # print(info)

            if args.use_risk and args.fine_tune_risk and len(rb) > 0 and global_step % args.risk_update_period == 0:
                    risk_data = rb.sample(args.risk_batch_size)
                    pred = risk_model(risk_data["next_obs"].to(device))
                    risk_loss = risk_criterion(pred, torch.argmax(risk_data["risks"].squeeze(), axis=1).to(device))
                    opt_risk.zero_grad()
                    risk_loss.backward()
                    opt_risk.step()
                    logger.store(**{"risk/risk_loss": risk_loss.item()})


            if "final_observation" in info:
                info["final_observation"] = np.array(
                    [
                        array if array is not None else np.zeros(obs.shape[-1])
                        for array in info["final_observation"]
                    ],
                )
                info["final_observation"] = torch.as_tensor(
                    info["final_observation"],
                    dtype=torch.float32,
                    device=device,
                )
                if args.use_risk and args.fine_tune_risk:
                    f_risks = torch.empty_like(f_costs)
                    for i in range(args.num_envs):
                        f_risks[:, i] = compute_fear(f_costs[:, i])
                    f_risks = f_risks.view(-1, 1)
                    f_risks_quant = torch.Tensor(np.apply_along_axis(lambda x: np.histogram(x, bins=risk_bins)[0], 1, np.expand_dims(f_risks.cpu().numpy(), 1)))
                    rb.add(None, f_next_obs.view(-1, obs_space.shape[0]), None, None, None, None, f_risks_quant, f_risks)

                    f_next_obs, f_costs = None, None
                final_risk = risk_model(info["final_observation"]) if args.use_risk else None


            buffer.store(
                obs=obs,
                act=act,
                reward=reward,
                cost=cost,
                value_r=value_r,
                value_c=value_c,
                log_prob=log_prob,
            )

            obs = next_obs
            risk = risk_model(obs) if args.use_risk else None
            epoch_end = steps >= local_steps_per_epoch - 1
            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1, device=device)
                    last_value_c = torch.zeros(1, device=device)
                    if not done:
                        if epoch_end:
                            with torch.no_grad():
                                risk_idx = risk[idx] if args.use_risk else None
                                _, _, last_value_r, last_value_c = policy.step(
                                    obs[idx], risk_idx, deterministic=False
                                )
                        if time_out:
                            with torch.no_grad():
                                final_risk_idx = final_risk[idx] if args.use_risk else None 
                                _, _, last_value_r, last_value_c = policy.step(
                                    info["final_observation"][idx], final_risk_idx, deterministic=False
                                )
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)
                    if done or time_out:
                        rew_deque.append(ep_ret[idx])
                        cost_deque.append(ep_cost[idx])
                        len_deque.append(ep_len[idx])
                        success_deque.append(ep_success[idx])
                        #goal_deque.append(info["final_info"][idx]["cum_goal_met"])
                        total_cost += ep_cost[idx]
                        violations = np.sum(np.array(cost_deque) > args.cost_limit)
                        total_violations += int(ep_cost[idx] > args.cost_limit)
                        logger.store(
                            **{
                                "Metrics/EpRet": np.mean(rew_deque),
                                "Metrics/EpRetStd": np.std(rew_deque),
                                "Metrics/EpCost": np.mean(cost_deque),
                                "Metrics/EpCostStd": np.std(cost_deque),
                                "Metrics/EpLen": np.mean(len_deque),
                                "Metrics/EpSuccess": np.mean(success_deque),
                                "Metrics/EpSuccessStd": np.std(success_deque),
                                #"Metrics/EpGoal": np.mean(goal_deque),
                                "Metrics/TotalCost": total_cost,
                                "Metrics/ViolationRate": np.mean(np.array(cost_deque) > args.cost_limit),
                                "Metrics/TotalViolation": total_violations,
                            }
                        )
                        ep_ret[idx] = 0.0
                        ep_cost[idx] = 0.0
                        ep_len[idx] = 0.0
                        ep_success[idx] = 0.0
                        logger.logged = False

                    buffer.finish_path(
                        last_value_r=last_value_r, last_value_c=last_value_c, idx=idx
                    )
        rollout_end_time = time.time()
        if epoch % 20 == 0 and args.eval_critic_performance:
            # Evaluate critic performance using fresh rollouts
            critic_metrics = evaluate_critic_performance_from_rollouts(
                args=args,
                policy=policy,
                env=env,
                num_episodes=int(100 / args.num_envs),
                max_ep_len=1000,  # Maximum episode length
                device=device,
                gamma=config['gamma'], 
                use_risk=args.use_risk,
                risk_model=risk_model if args.use_risk else None,
                create_plots=True
            )

            # Log the critic evaluation metrics
            logger.store(
                **{
                    # Reward critic metrics
                    "Reward Value/EstimationError": critic_metrics['reward_critic']['mean_error'],
                    "Reward Value/MeanAbsError": critic_metrics['reward_critic']['mean_abs_error'],
                    "Reward Value/OverestimationRatio": critic_metrics['reward_critic']['overestimation_ratio'],
                    "Reward Value/UnderestimationRatio": critic_metrics['reward_critic']['underestimation_ratio'],
                    "Reward Value/MaxError": critic_metrics['reward_critic']['max_error'],
                    "Reward Value/PearsonCorr": critic_metrics['reward_critic']['pearson_corr'],
                    "Reward Value/SpearmanCorr": critic_metrics['reward_critic']['spearman_corr'],
                    "Reward Value/KendallCorr": critic_metrics['reward_critic']['kendall_corr'],
                    "Reward Value/MeanPredicted": critic_metrics['reward_critic']['mean_value'],
                    "Reward Value/StdPredicted": critic_metrics['reward_critic']['std_value'],
                    "Reward Value/MinPredicted": critic_metrics['reward_critic']['min_value'],
                    "Reward Value/MaxPredicted": critic_metrics['reward_critic']['max_value'],
                    "Reward Value/MeanMCReturn": critic_metrics['reward_critic']['mean_mc_return'],
                    "Reward Value/StdMCReturn": critic_metrics['reward_critic']['std_mc_return'],
                    "Reward Value/MinMCReturn": critic_metrics['reward_critic']['min_mc_return'],
                    "Reward Value/MaxMCReturn": critic_metrics['reward_critic']['max_mc_return'],
                    
                    # Cost critic metrics
                    "Cost Value/EstimationError": critic_metrics['cost_critic']['mean_error'],
                    "Cost Value/MeanAbsError": critic_metrics['cost_critic']['mean_abs_error'],
                    "Cost Value/OverestimationRatio": critic_metrics['cost_critic']['overestimation_ratio'],
                    "Cost Value/UnderestimationRatio": critic_metrics['cost_critic']['underestimation_ratio'],
                    "Cost Value/MaxError": critic_metrics['cost_critic']['max_error'],
                    "Cost Value/PearsonCorr": critic_metrics['cost_critic']['pearson_corr'],
                    "Cost Value/SpearmanCorr": critic_metrics['cost_critic']['spearman_corr'],
                    "Cost Value/KendallCorr": critic_metrics['cost_critic']['kendall_corr'],
                    "Cost Value/MeanPredicted": critic_metrics['cost_critic']['mean_value'],
                    "Cost Value/StdPredicted": critic_metrics['cost_critic']['std_value'],
                    "Cost Value/MinPredicted": critic_metrics['cost_critic']['min_value'],
                    "Cost Value/MaxPredicted": critic_metrics['cost_critic']['max_value'],
                    "Cost Value/MeanMCReturn": critic_metrics['cost_critic']['mean_mc_return'],
                    "Cost Value/StdMCReturn": critic_metrics['cost_critic']['std_mc_return'],
                    "Cost Value/MinMCReturn": critic_metrics['cost_critic']['min_mc_return'],
                    "Cost Value/MaxMCReturn": critic_metrics['cost_critic']['max_mc_return'],
                }
            )

            # Log plots to wandb
            if 'plot_fig' in critic_metrics['reward_critic']:
                # Convert matplotlib figure to image
                reward_fig = critic_metrics['reward_critic']['plot_fig']
                reward_img = wandb.Image(reward_fig)
                wandb.log({"plots/reward_value_scatter": reward_img})
                plt.close(reward_fig)
            
            if 'plot_fig' in critic_metrics['cost_critic']:
                # Convert matplotlib figure to image
                cost_fig = critic_metrics['cost_critic']['plot_fig']
                cost_img = wandb.Image(cost_fig)
                wandb.log({"plots/cost_value_scatter": cost_img})
                plt.close(cost_fig)
        eval_start_time = time.time()

        eval_episodes = 1 if epoch < epochs - 1 else 10
        if args.use_eval:
            for _ in range(eval_episodes):
                eval_done = False
                eval_obs, _ = eval_env.reset()
                eval_obs = torch.as_tensor(eval_obs, dtype=torch.float32, device=device)
                eval_rew, eval_cost, eval_len = 0.0, 0.0, 0.0
                while not eval_done:
                    with torch.no_grad():
                        # if args.use_risk:
                        risk = risk_model(eval_obs) if args.use_risk else None
                        act, log_prob, value_r, value_c = policy.step(eval_obs, risk, deterministic=True)
                    next_obs, reward, cost, terminated, truncated, info = env.step(
                        act.detach().squeeze().cpu().numpy()
                    )
                    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
                    eval_rew += reward
                    eval_cost += cost
                    eval_len += 1
                    eval_done = terminated[0] or truncated[0]
                    eval_obs = next_obs
                eval_rew_deque.append(eval_rew)
                eval_cost_deque.append(eval_cost)
                eval_len_deque.append(eval_len)
                eval_goal_deque.append(info["final_info"][idx]["cum_goal_met"])
                eval_total_cost += eval_cost
            logger.store(
                **{
                    "Metrics/EvalEpRet": np.mean(eval_rew),
                    "Metrics/EvalEpCost": np.mean(eval_cost),
                    "Metrics/EvalEpLen": np.mean(eval_len),
                    "Metrics/EvalEpGoal": np.mean(eval_goal_deque),
                    "Metrics/EvalTotalCost": eval_total_cost,
                }
            )

        if epoch % 100 == 0 and args.record_gif:
            eval_env.obs_rms = env.obs_rms
            render_and_save_gif(eval_env, policy, device, max_steps=1000, use_risk=args.use_risk, risk_model=risk_model if args.use_risk else None, gif_name=f"episode_{epoch}", wandb_log=True)


        eval_end_time = time.time()

        # update lagrange multiplier
        ep_costs = logger.get_stats("Metrics/EpCost")
        lagrange.update_lagrange_multiplier(ep_costs)

        # update policy
        data = buffer.get()
        with torch.no_grad():
            data["risk"] = risk_model(data["obs"]) if args.use_risk else None
        fvp_obs = data["obs"][:: 1]
        fvp_risk = data["risk"][:: 1] if args.use_risk else None
        
        data["fvp_obs"] = fvp_obs
        data["fvp_risk"] = fvp_risk
        # Store old distribution and parameters before any updates
        old_distribution = policy.actor(data["obs"], data["risk"])
        theta_old = get_flat_params_from(policy.actor)
        assert theta_old is not None, "theta_old is None after initialization"
        policy.actor.zero_grad()

        # compute advantage
        advantage = data["adv_r"] - lagrange.lagrangian_multiplier * data["adv_c"]
        advantage /= (lagrange.lagrangian_multiplier + 1)
        
        # Compute initial loss before any updates
        if args.use_sam_actor:
            with torch.no_grad():
                temp_distribution = policy.actor(data["obs"], data["risk"])
                log_prob = temp_distribution.log_prob(data["act"]).sum(dim=-1)
                ratio = torch.exp(log_prob - data["log_prob"])
                loss_before = -(ratio * advantage).mean().item()
            
            # Get SAM gradients at perturbed point
            sam_grads, perturbed_params, cos_sim, effective_rho, scale_along_grad = actor_sam_fn(args)(
                fvp, policy, data, advantage, data["adv_c"], data["adv_r"],
                rho=args.sam_rho, 
                target_kl=args.perturbation_target_kl if not args.perturbation_decay else args.perturbation_target_kl / np.sqrt(epoch + 1),
                num_samples=args.sam_num_samples,
            )
            # Use SAM gradients for TRPO update
            x = conjugate_gradients(fvp, policy, fvp_obs, fvp_risk, -sam_grads, CONJUGATE_GRADIENT_ITERS)
            assert torch.isfinite(x).all(), "x is not finite"
            xHx = torch.dot(x, fvp(x, policy, fvp_obs, fvp_risk))
            assert xHx.item() >= 0, "xHx is negative"
            alpha = torch.sqrt(2 * config['target_kl'] / (xHx + 1e-8))
            step_direction = x * alpha
            assert torch.isfinite(step_direction).all(), "step_direction is not finite"
            grads = -sam_grads

        else:
            temp_distribution = policy.actor(data["obs"], data["risk"])
            log_prob = temp_distribution.log_prob(data["act"]).sum(dim=-1)
            ratio = torch.exp(log_prob - data["log_prob"])
            loss_pi = -(ratio * advantage).mean()
            loss_before = loss_pi.item()
            old_distribution = policy.actor(data["obs"], data["risk"])

            loss_pi.backward()

            grads = -get_flat_gradients_from(policy.actor)
            x = conjugate_gradients(fvp, policy, fvp_obs, fvp_risk, grads, CONJUGATE_GRADIENT_ITERS)
            assert torch.isfinite(x).all(), "x is not finite"
            xHx = torch.dot(x, fvp(x, policy, fvp_obs, fvp_risk))
            assert xHx.item() >= 0, "xHx is negative"
            alpha = torch.sqrt(2 * config['target_kl'] / (xHx + 1e-8))
            step_direction = x * alpha
            assert torch.isfinite(step_direction).all(), "step_direction is not finite"
        step_frac = 1.0
        # Change expected objective function gradient = expected_imrpove best this moment
        expected_improve = grads.dot(step_direction)

        final_kl = 0.0

        # While not within_trust_region and not out of total_steps:
        for step in range(TRPO_SEARCHING_STEPS):
            # update theta params
            new_theta = theta_old + step_frac * step_direction
            # set new params as params of net
            set_param_values_to_model(policy.actor, new_theta)

            with torch.no_grad():
                temp_distribution = policy.actor(data["obs"], data["risk"])
                log_prob = temp_distribution.log_prob(data["act"]).sum(dim=-1)
                ratio = torch.exp(log_prob - data["log_prob"])
                loss_pi = -(ratio * advantage).mean()
                
                # compute KL distance between new and old policy
                current_distribution = policy.actor(data["obs"], data["risk"])
                kl = (
                    torch.distributions.kl.kl_divergence(
                        old_distribution, current_distribution
                    )
                    .mean()
                    .item()
                )
            
            # real loss improve: old policy loss - new policy loss
            loss_improve = loss_before - loss_pi.item()
            logger.log(
                f"Expected Improvement: {expected_improve} Actual: {loss_improve}"
            )
            if not torch.isfinite(loss_pi):
                logger.log("WARNING: loss_pi not finite")
            elif loss_improve < 0:
                logger.log("INFO: did not improve improve <0")
            elif kl > config["target_kl"]:
                logger.log("INFO: violated KL constraint.")
            else:
                # step only if surrogate is improved and when within trust reg.
                acceptance_step = step + 1
                logger.log(f"Accept step at i={acceptance_step}")
                final_kl = kl
                break
            step_frac *= 0.8
        else:
            logger.log("INFO: no suitable step found...")
            step_direction = torch.zeros_like(step_direction)
            acceptance_step = 0

        theta_new = theta_old + step_frac * step_direction
        set_param_values_to_model(policy.actor, theta_new)

        logger.store(
            **{
                "Misc/Alpha": alpha.item(),
                "Misc/FinalStepNorm": torch.norm(step_direction).mean().item(),
                "Misc/xHx": xHx.item(),
                "Misc/gradient_norm": torch.norm(grads).mean().item(),
                "Misc/H_inv_g": x.norm().item(),
                "Misc/AcceptanceStep": acceptance_step,
                "Loss/Loss_actor": loss_pi.mean().item(),
                "Train/KL": final_kl,
            },
        )
        if args.use_sam_actor and cos_sim is not None and effective_rho is not None and scale_along_grad is not None:
            logger.store(
                **{
                    "Misc/CosineSimilarity": cos_sim,
                    "Misc/EffectiveRho": effective_rho,
                    "Misc/ScaleAlongGrad": scale_along_grad,
                }
            )
        dataloader = DataLoader(
            dataset=TensorDataset(
                data["obs"],
                data["risk"] if args.use_risk else data["obs"],
                data["target_value_r"],
                data["target_value_c"],
            ),
            batch_size=config.get("batch_size", args.steps_per_epoch//config.get("num_mini_batch", 1)),
            shuffle=True,
        )
        for _ in range(config["learning_iters"]):
            for (
                obs_b,
                risk_b,
                target_value_r_b,
                target_value_c_b,
            ) in dataloader:
                risk_b = risk_b if args.use_risk else None
                
                # Update reward critic
                reward_critic_optimizer.zero_grad()
                if args.use_sam_reward_critic:
                    if args.sam_type == "v4":
                        sam_grads_r, _ = compute_sam_gradients_critic_v4(
                            policy.reward_critic, 
                            {"obs": obs_b, "risk": risk_b}, 
                            target_value_r_b,
                            rho=args.sam_rho, 
                            num_samples=args.sam_num_samples
                        )
                    else:
                        sam_grads_r, _, _, _, _ = compute_sam_gradients_critic(
                            policy.reward_critic, 
                            {"obs": obs_b, "risk": risk_b}, 
                            target_value_r_b,
                            rho=args.sam_rho)
                    for name, param in policy.reward_critic.named_parameters():
                        if name in sam_grads_r:
                            param.grad = sam_grads_r[name]
                    if config.get("use_critic_norm", True):
                        for param in policy.reward_critic.parameters():
                            if param.grad is not None:
                                param.grad += param * 0.001
                    clip_grad_norm_(policy.reward_critic.parameters(), config["max_grad_norm"])
                    reward_critic_optimizer.step()
                else:
                    # Standard reward critic update
                    loss_r = nn.functional.mse_loss(policy.reward_critic(obs_b, risk_b), target_value_r_b)
                    if config.get("use_critic_norm", True):
                        for param in policy.reward_critic.parameters():
                            loss_r += param.pow(2).sum() * 0.001
                    loss_r.backward()
                    clip_grad_norm_(policy.reward_critic.parameters(), config["max_grad_norm"])
                    reward_critic_optimizer.step()
                
                # Update cost critic
                cost_critic_optimizer.zero_grad()
                if args.use_sam_cost_critic:
                    if args.sam_type == "v4":
                        sam_grads_c, _ = compute_sam_gradients_critic_v4(
                            policy.cost_critic, 
                            {"obs": obs_b, "risk": risk_b}, 
                            target_value_c_b,
                            rho=args.sam_rho,
                            num_samples=args.sam_num_samples
                        )
                    else:
                        sam_grads_c, _, _, _, _ = compute_sam_gradients_critic(
                            policy.cost_critic, 
                            {"obs": obs_b, "risk": risk_b}, 
                            target_value_c_b,
                            rho=args.sam_rho)
                    for name, param in policy.cost_critic.named_parameters():
                        if name in sam_grads_c:
                            param.grad = sam_grads_c[name]
                    if config.get("use_critic_norm", True):
                        for param in policy.cost_critic.parameters():
                            if param.grad is not None:
                                param.grad += param * 0.001
                    clip_grad_norm_(policy.cost_critic.parameters(), config["max_grad_norm"])
                    cost_critic_optimizer.step()
                else:
                    # Standard cost critic update
                    loss_c = nn.functional.mse_loss(policy.cost_critic(obs_b, risk_b), target_value_c_b)
                    if config.get("use_critic_norm", True):
                        for param in policy.cost_critic.parameters():
                            loss_c += param.pow(2).sum() * 0.001
                    loss_c.backward()
                    clip_grad_norm_(policy.cost_critic.parameters(), config["max_grad_norm"])
                    cost_critic_optimizer.step()

                # Compute losses for logging (always compute for logging purposes)
                with torch.no_grad():
                    value_r = policy.reward_critic(obs_b, risk_b)
                    value_c = policy.cost_critic(obs_b, risk_b)
                    loss_r = nn.functional.mse_loss(value_r, target_value_r_b)
                    loss_c = nn.functional.mse_loss(value_c, target_value_c_b)
                
                logger.store(
                    **{
                        "Loss/Loss_reward_critic": loss_r.mean().item(),
                        "Loss/Loss_cost_critic": loss_c.mean().item(),
                    }
                )
        update_end_time = time.time()
        torch.save(policy.state_dict(), os.path.join(wandb.run.dir, "policy.pt"))
        wandb.save("policy.pt")
        if args.use_risk:
            torch.save(risk_model.state_dict(), os.path.join(args.log_dir, "risk_model.pt"))
            wandb.save(os.path.join(args.log_dir, "risk_model.pt"))



        if not logger.logged:
            # log data
            logger.log_tabular("Metrics/EpRet")
            logger.log_tabular("Metrics/EpCost")
            logger.log_tabular("Metrics/TotalCost")
            logger.log_tabular("Metrics/EpLen")
            logger.log_tabular("Metrics/EpSuccess")
            logger.log_tabular("Metrics/EpSuccessStd")
            logger.log_tabular("Metrics/EpRetStd")
            logger.log_tabular("Metrics/EpCostStd")
            #logger.log_tabular("Metrics/EpGoal")
            if args.use_eval:
                logger.log_tabular("Metrics/EvalEpRet")
                logger.log_tabular("Metrics/EvalEpCost")
                logger.log_tabular("Metrics/EvalEpLen")
                logger.log_tabular("Metrics/EvalTotalCost")
                logger.log_tabular("Metrics/EvalEpGoal")

            logger.log_tabular("Train/Epoch", epoch + 1)
            logger.log_tabular("Train/TotalSteps", (epoch + 1) * args.steps_per_epoch)
            logger.log_tabular("Train/KL")
            logger.log_tabular("Train/LagragianMultiplier", lagrange.lagrangian_multiplier)
            logger.log_tabular("Loss/Loss_reward_critic")
            logger.log_tabular("Loss/Loss_cost_critic")
            logger.log_tabular("Loss/Loss_actor")
            logger.log_tabular("Time/Rollout", rollout_end_time - rollout_start_time)
            if args.use_eval:
                logger.log_tabular("Time/Eval", eval_end_time - eval_start_time)
            logger.log_tabular("Time/Update", update_end_time - eval_end_time)
            logger.log_tabular("Time/Total", update_end_time - rollout_start_time)
            logger.log_tabular("Value/RewardAdv", data["adv_r"].mean().item())
            logger.log_tabular("Value/CostAdv", data["adv_c"].mean().item())
            logger.log_tabular("Misc/Alpha")
            logger.log_tabular("Misc/FinalStepNorm")
            logger.log_tabular("Misc/xHx")
            logger.log_tabular("Misc/gradient_norm")
            logger.log_tabular("Misc/H_inv_g")
            logger.log_tabular("Misc/AcceptanceStep")
            logger.log_tabular("Metrics/ViolationRate")
            logger.log_tabular("Metrics/TotalViolation")
            if epoch % 20 == 0 and args.eval_critic_performance:
                # Add critic evaluation metrics
                logger.log_tabular("Reward Value/EstimationError")
                logger.log_tabular("Reward Value/MeanAbsError") 
                logger.log_tabular("Reward Value/OverestimationRatio")
                logger.log_tabular("Reward Value/UnderestimationRatio")
                logger.log_tabular("Reward Value/MaxError")
                logger.log_tabular("Reward Value/PearsonCorr")
                logger.log_tabular("Reward Value/SpearmanCorr")
                logger.log_tabular("Reward Value/KendallCorr")
                logger.log_tabular("Reward Value/MeanPredicted")
                logger.log_tabular("Reward Value/StdPredicted")
                logger.log_tabular("Reward Value/MeanMCReturn")
                logger.log_tabular("Reward Value/StdMCReturn")
                logger.log_tabular("Reward Value/MinMCReturn")
                logger.log_tabular("Reward Value/MaxMCReturn")
                logger.log_tabular("Reward Value/MinPredicted")
                logger.log_tabular("Reward Value/MaxPredicted")
                logger.log_tabular("Cost Value/EstimationError")
                logger.log_tabular("Cost Value/MeanAbsError")
                logger.log_tabular("Cost Value/OverestimationRatio")
                logger.log_tabular("Cost Value/UnderestimationRatio")
                logger.log_tabular("Cost Value/MaxError")
                logger.log_tabular("Cost Value/PearsonCorr")
                logger.log_tabular("Cost Value/SpearmanCorr")
                logger.log_tabular("Cost Value/KendallCorr")
                logger.log_tabular("Cost Value/MeanPredicted")
                logger.log_tabular("Cost Value/StdPredicted")
                logger.log_tabular("Cost Value/MinPredicted")
                logger.log_tabular("Cost Value/MaxPredicted")
                logger.log_tabular("Cost Value/MeanMCReturn")
                logger.log_tabular("Cost Value/StdMCReturn")
                logger.log_tabular("Cost Value/MinMCReturn")
                logger.log_tabular("Cost Value/MaxMCReturn")

            if args.use_sam_actor and cos_sim is not None and effective_rho is not None and scale_along_grad is not None:
                logger.log_tabular("Misc/CosineSimilarity")
                logger.log_tabular("Misc/EffectiveRho")
                logger.log_tabular("Misc/ScaleAlongGrad")
            if args.use_risk and args.fine_tune_risk:
                #try:
                logger.log_tabular("risk/risk_loss")
                #except:
                #    pass
            logger.dump_tabular()
            if (epoch+1) % 100 == 0 or epoch == 0:
                logger.torch_save(itr=epoch)
                # if args.task not in isaac_gym_map.keys():
                #     logger.save_state(
                #         state_dict={
                #             "Normalizer": env.obs_rms,
                #         },
                #         itr = epoch
                #     )
        ## Garbage Collection 
        data, dataloader = None, None
    ## Save Policy 
    torch.save(policy.state_dict(), os.path.join(args.log_dir, "policy.pt"))
    wandb.save(os.path.join(args.log_dir, "policy.pt"))
    if args.use_risk:
        torch.save(risk_model.state_dict(), os.path.join(args.log_dir, "risk_model.pt"))
        wandb.save(os.path.join(args.log_dir, "risk_model.pt"))
    logger.close()


if __name__ == "__main__":
    args, cfg_env = single_agent_args()
    import wandb
    run = wandb.init(config=vars(args), entity="liam-paull",
                project="sam-safe-rl",
                settings=wandb.Settings(_service_wait=60),
                # monitor_gym=True,
                sync_tensorboard=True, save_code=True)
    relpath = time.strftime("%Y-%m-%d-%H-%M-%S")
    subfolder = "-".join(["seed", str(args.seed).zfill(3)])
    relpath = "-".join([subfolder, relpath])
    algo = os.path.basename(__file__).split(".")[0]
    args.log_dir = os.path.join(args.log_dir, args.experiment, args.task, algo, run.name)
    if not args.write_terminal:
        terminal_log_name = "terminal.log"
        error_log_name = "error.log"
        terminal_log_name = f"seed{args.seed}_{terminal_log_name}"
        error_log_name = f"seed{args.seed}_{error_log_name}"
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
        with open(
            os.path.join(
                f"{args.log_dir}",
                terminal_log_name,
            ),
            "w",
            encoding="utf-8",
        ) as f_out:
            sys.stdout = f_out
            with open(
                os.path.join(
                    f"{args.log_dir}",
                    error_log_name,
                ),
                "w",
                encoding="utf-8",
            ) as f_error:
                sys.stderr = f_error
                main(args, cfg_env)
    else:
        main(args, cfg_env)
