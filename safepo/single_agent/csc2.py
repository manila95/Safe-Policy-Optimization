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
from safepo.common.env import make_sa_mujoco_env, make_sa_isaac_env
from safepo.common.lagrange import PIDLagrangian as Lagrange
from safepo.common.logger import EpochLogger
from safepo.common.model import ActorVQCritic
from safepo.utils.config import single_agent_args, isaac_gym_map, parse_sim_params
from src.models.risk_models import *
from src.datasets.risk_datasets import *
from src.utils import * 
from copy import deepcopy

CONJUGATE_GRADIENT_ITERS=15
TRPO_SEARCHING_STEPS=15

default_cfg = {
    'hidden_sizes': [64, 64],
    'gamma': 0.99,
    'target_kl': 0.01,
    'batch_size': 128,
    'learning_iters': 10,
    'max_grad_norm': 40.0,
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
}


def get_flat_params_from(model: torch.nn.Module) -> torch.Tensor:
    flat_params = []
    for _, param in model.named_parameters():
        if param.requires_grad:
            data = param.data
            data = data.view(-1)  # flatten tensor
            flat_params.append(data)
    assert flat_params, "No gradients were found in model parameters."
    return torch.cat(flat_params)


def conjugate_gradients(
    fisher_product: Callable[[torch.Tensor], torch.Tensor],
    policy: ActorVCritic,
    fvp_obs: torch.Tensor,
    fvp_risk: torch.Tensor,
    vector_b: torch.Tensor,
    num_steps: int = 10,
    residual_tol: float = 1e-10,
    eps: float = 1e-6,
) -> torch.Tensor:
    vector_x = torch.zeros_like(vector_b)
    vector_r = vector_b - fisher_product(vector_x, policy, fvp_obs, fvp_risk)
    vector_p = vector_r.clone()
    rdotr = torch.dot(vector_r, vector_r)

    for _ in range(num_steps):
        vector_z = fisher_product(vector_p, policy, fvp_obs, fvp_risk)
        alpha = rdotr / (torch.dot(vector_p, vector_z) + eps)
        vector_x += alpha * vector_p
        vector_r -= alpha * vector_z
        new_rdotr = torch.dot(vector_r, vector_r)
        if torch.sqrt(new_rdotr) < residual_tol:
            break
        vector_mu = new_rdotr / (rdotr + eps)
        vector_p = vector_r + vector_mu * vector_p
        rdotr = new_rdotr
    return vector_x


def set_param_values_to_model(model: torch.nn.Module, vals: torch.Tensor) -> None:
    assert isinstance(vals, torch.Tensor)
    i: int = 0
    for _, param in model.named_parameters():
        if param.requires_grad:  # param has grad and, hence, must be set
            orig_size = param.size()
            size = np.prod(list(param.size()))
            new_values = vals[i : int(i + size)]
            # set new param values
            new_values = new_values.view(orig_size)
            param.data = new_values
            i += int(size)  # increment array position
    assert i == len(vals), f"Lengths do not match: {i} vs. {len(vals)}"


def get_flat_gradients_from(model: torch.nn.Module) -> torch.Tensor:
    grads = []
    for _, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad = param.grad
            grads.append(grad.view(-1))  # flatten tensor and append
    assert grads, "No gradients were found in model parameters."
    return torch.cat(grads)


def fvp(
    params: torch.Tensor,
    policy: ActorVCritic,
    fvp_obs: torch.Tensor,
    fvp_risk: torch.Tensor,
) -> torch.Tensor:
    policy.actor.zero_grad()
    current_distribution = policy.actor(fvp_obs, fvp_risk)
    with torch.no_grad():
        old_distribution = policy.actor(fvp_obs, fvp_risk)
    kl = torch.distributions.kl.kl_divergence(
        old_distribution, current_distribution
    ).mean()

    grads = torch.autograd.grad(kl, tuple(policy.actor.parameters()), create_graph=True)
    flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

    kl_p = (flat_grad_kl * params).sum()
    grads = torch.autograd.grad(
        kl_p,
        tuple(policy.actor.parameters()),
        retain_graph=False,
    )

    flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads])

    return flat_grad_grad_kl + params * 0.1


def compute_loss_qc(ac, ac_targ, data, obs_dim, act_dim, gamma, loss_fn, alpha=-1.):
    o, a, c, o2, d = [data[s] for s in ('obs', 'act', 'cost', 'obs2', 'done')]
    # print(o.size(), a.size())
    q = ac.cost_critic(torch.cat([o, a], axis=-1))
    with torch.no_grad():
        pi_next = ac_targ.actor(o2)
        #pi_next = ac.pi._distribution(o2)
        acts_next = pi_next.sample_n(100)
        obs_next_flat = o2.unsqueeze(0).repeat(100, 1, 1).view(-1, obs_dim)
        acts_next_flat = acts_next.view(-1, act_dim)
        q_pi_targ_flat = torch.clamp(ac_targ.cost_critic(torch.cat([obs_next_flat, acts_next_flat], axis=-1)), 0, 1.0)
        #q_pi_targ_flat = torch.clamp(ac.qc(obs_next_flat, acts_next_flat),
        #                             *vc_range)
        q_pi_targ = q_pi_targ_flat.view(100, -1).mean(0)
        backup = c + gamma*(1-d)*q_pi_targ

    # Conservative regularization
    with torch.no_grad():
        pi = ac.actor(o)
        acts = pi.sample_n(100)
        obs_flat = o.unsqueeze(0).repeat(100, 1, 1).view(-1, obs_dim)
        acts_flat = acts.view(-1, act_dim)
    q_pi = ac.cost_critic(torch.cat([obs_flat, acts_flat], -1)).view(100, -1).mean(0)

    loss = loss_fn(q, backup) #+ alpha*(q-q_pi).mean()
    #loss = loss_fn(q, backup) - alpha*q_pi.mean()
    loss_info = dict(QcVals=q.detach().numpy())

    return loss, loss_info


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
        env, obs_space, act_space = make_sa_mujoco_env(args,
            num_envs=args.num_envs, env_id=args.task, seed=args.seed
        )
        eval_env, _, _ = make_sa_mujoco_env(args, num_envs=1, env_id=args.task, seed=None)
        config = default_cfg

    else:
        sim_params = parse_sim_params(args, cfg_env, None)
        env = make_sa_isaac_env(args=args, cfg=cfg_env, sim_params=sim_params)
        eval_env = env
        obs_space = env.observation_space
        act_space = env.action_space
        args.num_envs = env.num_envs
        config = isaac_gym_specific_cfg
    obs_dim=obs_space.shape[0]
    act_dim=act_space.shape[0]
    # set training steps
    steps_per_epoch = config.get("steps_per_epoch", args.steps_per_epoch)
    total_steps = config.get("total_steps", args.total_steps)
    local_steps_per_epoch = steps_per_epoch // args.num_envs
    epochs = total_steps // steps_per_epoch
    # create the actor-critic module
    policy = ActorVQCritic(
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        hidden_sizes=config["hidden_sizes"],
        use_risk=args.use_risk,
        risk_size=risk_size,
    ).to(device)
    reward_critic_optimizer = torch.optim.Adam(
        policy.reward_critic.parameters(), lr=1e-3
    )
    cost_critic_optimizer = torch.optim.Adam(
        policy.cost_critic.parameters(), lr=1e-3
    )
    policy_target = deepcopy(policy)
    loss_fn = torch.nn.SmoothL1Loss()

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
            rb = ReplayBuffer(buffer_size=args.total_steps)

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
    eval_rew_deque = deque(maxlen=50)
    eval_cost_deque = deque(maxlen=50)
    eval_len_deque = deque(maxlen=50)
    eval_goal_deque = deque(maxlen=50)
    logger.save_config(dict_args)
    logger.setup_torch_saver(policy.actor)
    logger.log("Start with training.")
    obs, _ = env.reset()
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    ep_ret, ep_cost, ep_len, ep_goal = (
        np.zeros(args.num_envs),
        np.zeros(args.num_envs),
        np.zeros(args.num_envs),
        np.zeros(args.num_envs),
    )
    total_cost, eval_total_cost = 0, 0
    f_next_obs, f_costs = None, None
    global_step = 0
    # training loop
    for epoch in range(epochs):
        rollout_start_time = time.time()
        # collect samples until we have enough to update
        for steps in range(local_steps_per_epoch):
            with torch.no_grad():
                    risk = risk_model(obs) if args.use_risk else None
                    act, log_prob, value_r, value_c = policy.step(obs, risk, deterministic=False)

            action = act.detach().squeeze() if args.task in isaac_gym_map.keys() else act.detach().squeeze().cpu().numpy()
            next_obs, reward, cost, terminated, truncated, info = env.step(action)

            ep_ret += reward.cpu().numpy() if args.task in isaac_gym_map.keys() else reward
            ep_cost += cost.cpu().numpy() if args.task in isaac_gym_map.keys() else cost
            ep_len += 1
            next_obs, reward, cost, terminated, truncated = (
                torch.as_tensor(x, dtype=torch.float32, device=device)
                for x in (next_obs, reward, cost, terminated, truncated)
            )
            done = np.logical_or(terminated, truncated)
    
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
                    e_risks_quant = torch.Tensor(np.apply_along_axis(lambda x: np.histogram(x, bins=risk_bins)[0], 1, f_risks.cpu().numpy())).to(device)
                    rb.add(None, f_next_obs.view(-1, obs_space.shape[0]), None, None, None, None, e_risks_quant, f_risks)
                    f_next_obs, f_costs = None, None
                final_risk = risk_model(info["final_observation"]) if args.use_risk else None
            global_step += args.num_envs


            buffer.store(
                obs=obs,
                next_obs=next_obs,
                act=act,
                reward=reward,
                done=done,
                cost=cost,
                value_r=value_r,
                value_c=value_c,
                log_prob=log_prob,
            )
            if args.use_risk and args.fine_tune_risk:
                if args.risk_input == "state_action":
                    obs_action = torch.cat([obs, action], axis=-1)
                    next_obs_action = torch.cat([next_obs, torch.zeros_like(action)], axis=-1)
                    f_next_obs = obs_action.unsqueeze(0) if f_next_obs is None else torch.concat([f_next_obs, obs_action.unsqueeze(0)], axis=0)
                    f_next_obs = next_obs_action.unsqueeze(0) if f_next_obs is None else torch.concat([f_next_obs, next_obs_action.unsqueeze(0)], axis=0)
                    f_costs = cost.unsqueeze(0) if f_costs is None else torch.concat([f_costs, cost.unsqueeze(0)], axis=0)
                    f_costs = cost.unsqueeze(0) if f_costs is None else torch.concat([f_costs, cost.unsqueeze(0)], axis=0)
                else:
                    f_next_obs = next_obs.unsqueeze(0) if f_next_obs is None else torch.concat([f_next_obs, next_obs.unsqueeze(0)], axis=0)
                    f_costs = cost.unsqueeze(0) if f_costs is None else torch.concat([f_costs, cost.unsqueeze(0)], axis=0)
            # print(info)
            if args.use_risk and args.fine_tune_risk:
                if len(rb) > args.risk_batch_size and global_step % args.risk_update_period == 0:
                    # for _ in range(args.num_risk_epochs):
                    risk_data = rb.sample(args.risk_batch_size)
                    risk_loss = risk_update_step(risk_model, risk_data, risk_criterion, opt_risk, device)
                    logger.store(**{"risk/risk_loss": risk_loss.item()})
                else:
                    logger.store(**{"risk/risk_loss": 0})



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
                        # last_value_r = last_value_r.unsqueeze(0)
                        # last_value_c = last_value_c.unsqueeze(0)
                    if done or time_out:
                        rew_deque.append(ep_ret[idx])
                        cost_deque.append(ep_cost[idx])
                        len_deque.append(ep_len[idx])
                        goal_deque.append(info["final_info"][idx]["cum_goal_met"])
                        total_cost += ep_cost[idx]
                        logger.store(
                            **{
                                "Metrics/EpRet": np.mean(rew_deque),
                                "Metrics/EpCost": np.mean(cost_deque),
                                "Metrics/EpLen": np.mean(len_deque),
                                "Metrics/EpGoal": np.mean(goal_deque),
                                "Metrics/TotalCost": total_cost,
                            }
                        )
                        ep_ret[idx] = 0.0
                        ep_cost[idx] = 0.0
                        ep_len[idx] = 0.0
                        logger.logged = False

                    buffer.finish_path(
                        last_value_r=last_value_r, last_value_c=last_value_c, idx=idx
                    )
        rollout_end_time = time.time()

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
        theta_old = get_flat_params_from(policy.actor)
        policy.actor.zero_grad()

        # comnpute advantage
        advantage = data["adv_r"] - lagrange.lagrangian_multiplier * data["adv_c"]
        advantage /= (lagrange.lagrangian_multiplier + 1)

        # compute loss_pi
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

        dataloader = DataLoader(
            dataset=TensorDataset(
                data["obs"],
                data["risk"] if args.use_risk else data["obs"],
                data["act"],
                data["next_obs"],
                data["done"],
                data["cost"],
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
                act_b,
                next_obs_b,
                done_b,
                cost_b,
                target_value_r_b,
                target_value_c_b,
            ) in dataloader:
                risk_b = risk_b if args.use_risk else None
                reward_critic_optimizer.zero_grad()
                loss_r = nn.functional.mse_loss(policy.reward_critic(obs_b, risk_b), target_value_r_b)
                cost_critic_optimizer.zero_grad()

                csc_data = {}
                csc_data["obs"], csc_data["act"], csc_data["obs2"], csc_data["done"], csc_data["cost"] = obs_b, act_b, next_obs_b, done_b, cost_b
                loss_c, _ = compute_loss_qc(policy, policy_target, csc_data, obs_dim, act_dim, config["gamma"], loss_fn)
                if config.get("use_critic_norm", True):
                    for param in policy.reward_critic.parameters():
                        loss_r += param.pow(2).sum() * 0.001
                    for param in policy.cost_critic.parameters():
                        loss_c += param.pow(2).sum() * 0.001
                total_loss = 2*loss_r + loss_c \
                    if config.get("use_value_coefficient", False) \
                    else loss_r + loss_c
                total_loss.backward()
                clip_grad_norm_(policy.parameters(), config["max_grad_norm"])
                reward_critic_optimizer.step()
                cost_critic_optimizer.step()

                logger.store(
                    **{
                        "Loss/Loss_reward_critic": loss_r.mean().item(),
                        "Loss/Loss_cost_critic": loss_c.mean().item(),
                    }
                )
    
        if epoch % 1 == 0:
            for param, target_param in zip(policy.cost_critic.parameters(), policy_target.cost_critic.parameters()):
                target_param.data.copy_(0.1 * param.data + (1 - 0.1) * target_param.data)
                
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
            logger.log_tabular("Metrics/EpGoal")
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
            if args.use_risk and args.fine_tune_risk:
                #try:
                logger.log_tabular("risk/risk_loss")
                #except:
                #    pass
            logger.dump_tabular()
            if (epoch+1) % 100 == 0 or epoch == 0:
                logger.torch_save(itr=epoch)
                if args.task not in isaac_gym_map.keys():
                    logger.save_state(
                        state_dict={
                            "Normalizer": env.obs_rms,
                        },
                        itr = epoch
                    )
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
    import os
    try:
        os.makedirs(os.path.join("/logs", args.experiment))
    except:
        pass
    run = wandb.init(config=vars(args), entity="manila95",
                project="risk_aware_exploration",
                monitor_gym=True,
                dir=os.path.join("/logs",args.experiment),
                sync_tensorboard=True, save_code=True)
    relpath = time.strftime("%Y-%m-%d-%H-%M-%S")
    subfolder = "-".join(["seed", str(args.seed).zfill(3)])
    relpath = "-".join([subfolder, relpath])
    algo = os.path.basename(__file__).split(".")[0]
    args.log_dir = wandb.run.dir #os.path.join(args.log_dir, args.experiment, args.task, algo, run.name)
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
