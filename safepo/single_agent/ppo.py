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

import numpy as np
try: 
    from isaacgym import gymutil
except ImportError:
    pass
import torch
import torch.nn as nn
import torch.optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, TensorDataset

from safepo.common.buffer import VectorizedOnPolicyBuffer
from safepo.common.env import make_sa_mujoco_env, make_sa_isaac_env
from safepo.common.logger import EpochLogger
from safepo.common.model import ActorVCritic
from safepo.utils.config import single_agent_args, isaac_gym_map, parse_sim_params

from src.models.risk_models import *
from src.datasets.risk_datasets import *
from src.utils import * 
import matplotlib.pyplot as plt


default_cfg = {
    'hidden_sizes': [64, 64],
    'gamma': 0.99,
    'target_kl': 0.02,
    'batch_size': 64,
    'learning_iters': 40,
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

def main(args, cfg_env=None):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    device = torch.device(f'{args.device}:{args.device_id}')
    torch.multiprocessing.set_start_method('spawn')# good solution !!!!

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
    ).to(device)
    actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=3e-4)
    actor_scheduler = LinearLR(
        actor_optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=epochs,
        verbose=False,
    )
    reward_critic_optimizer = torch.optim.Adam(
        policy.reward_critic.parameters(), lr=3e-4
    )
    cost_critic_optimizer = torch.optim.Adam(
        policy.cost_critic.parameters(), lr=3e-4
    )
    if os.path.exists(args.policy_model_path):
        policy.load_state_dict(torch.load(args.policy_model_path))#, map_location=device))
        print("Pretrained Policy loaded successfully")

    if args.use_risk:
        risk_model_class = {"bayesian": {"continuous": BayesRiskEstCont, "binary": BayesRiskEst, "quantile": BayesRiskEst}, 
                    "mlp": {"continuous": RiskEst, "binary": RiskEst}} 
         
        risk_input_shape = obs_space.shape[0]+act_space.shape[0] if args.risk_input == "state_action" else obs_space.shape[0]
        risk_model = BayesRiskEst(obs_size=risk_input_shape, batch_norm=True, out_size=risk_size)
        if os.path.exists(args.risk_model_path):
            risk_model.load_state_dict(torch.load(args.risk_model_path)) #, map_location=device))
            print("Pretrained Risk model loaded successfully")

        risk_model.to(device)
        risk_model.eval()

        opt_risk = torch.optim.Adam(risk_model.parameters(), lr=args.risk_lr, eps=1e-10)

        if args.fine_tune_risk:
            rb = ReplayBuffer(buffer_size=args.total_steps, fear_radius=args.fear_radius, device=device)

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
    ep_ret, ep_cost, ep_len = (
        np.zeros(args.num_envs),
        np.zeros(args.num_envs),
        np.zeros(args.num_envs),
    )
    total_cost, eval_total_cost = 0, 0
    f_next_obs, f_costs = None, None
    save_obs = obs[0]
    # risk_stats = np.zeros((epochs, risk_size))
    flag = 1
    # training loop
    for epoch in range(epochs):
        rollout_start_time = time.time()
        # collect samples until we have enough to update
        for steps in range(local_steps_per_epoch):
            with torch.no_grad():
                if args.use_risk:
                    if epoch < args.start_using_risk:
                        risk = torch.as_tensor(np.zeros((args.num_envs, risk_size)), dtype=torch.float32, device=device)
                    else:
                        risk = torch.exp(risk_model(torch.cat([obs, torch.zeros(args.num_envs, act_space.shape[0])], axis=-1))) if args.risk_input=="state_action" else torch.exp(risk_model(obs)) 
                else:
                    risk = None 
                act, log_prob, value_r, value_c = policy.step(obs, risk, deterministic=False)          
            action = act.detach().squeeze() if args.task in isaac_gym_map.keys() else act.detach().squeeze().cpu().numpy()
            next_obs, reward, cost, terminated, truncated, info = env.step(action)

            ep_ret += reward.cpu().numpy() if args.task in isaac_gym_map.keys() else reward
            ep_cost += cost.cpu().numpy() if args.task in isaac_gym_map.keys() else cost
            ep_len += 1
            next_obs, action, reward, cost, terminated, truncated = (
                torch.as_tensor(x, dtype=torch.float32, device=device)
                for x in (next_obs, action, reward, cost, terminated, truncated)
            )
            #print(obs.size(), next_obs.size(), action.size())
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
                        #if cost[i] > 0:
                        #    print(f_risks[:, i])
                    f_risks = f_risks.view(-1, 1)
                    
                    e_risks_quant = torch.Tensor(np.apply_along_axis(lambda x: np.histogram(x, bins=risk_bins)[0], 1, f_risks.cpu().numpy())).to(device)
                    rb.add(None, f_next_obs.view(-1, risk_input_shape), None, None, None, None, e_risks_quant, f_risks)
                    print(len(rb))
                    f_next_obs, f_costs = None, None

                if args.use_risk:
                    final_risk = torch.exp(risk_model(torch.cat([info["final_observation"], torch.zeros(args.num_envs, act_space.shape[0])], axis=-1))) if args.risk_input == "state_action" else torch.exp(risk_model(info["final_observation"]))
                else:
                    final_risk = None
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
            if args.use_risk:
                if epoch < args.start_using_risk:
                    risk = torch.zeros((obs.size()[0], risk_size))
                else:
                    risk = torch.exp(risk_model(torch.cat([obs, torch.zeros(args.num_envs, act_space.shape[0])], axis=-1))) if args.risk_input == "state_action" else torch.exp(risk_model(obs))
            else:
                risk = None

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
                        risk = torch.exp(risk_model(eval_obs)) if args.use_risk else None
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
                eval_total_cost += eval_cost
                eval_rew_deque.append(eval_rew)
                eval_cost_deque.append(eval_cost)
                eval_len_deque.append(eval_len)
                eval_goal_deque.append(info["final_info"][idx]["cum_goal_met"])

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


        ## Risk Fine Tuning before the policy is updated
        if args.use_risk and args.fine_tune_risk:
            if False:
                risk_data = rb.sample(args.num_risk_samples) if args.risk_update == "offline" else rb.slice_data(len(rb)-steps_per_epoch, len(rb))
                risk_dataset = RiskyDataset(risk_data["next_obs"].to(device), None, risk_data["risks"].to(device), False, risk_type=args.risk_type,
                                        fear_clip=None, fear_radius=args.fear_radius, one_hot=True, quantile_size=args.quantile_size, quantile_num=args.quantile_num)
                risk_dataloader = DataLoader(risk_dataset, batch_size=args.risk_batch_size, shuffle=True, num_workers=4, generator=torch.Generator(device="cpu"))

                risk_loss = train_risk(risk_model, risk_dataloader, risk_criterion, opt_risk, args.num_risk_epochs, device)
                logger.store(**{"risk/risk_loss": risk_loss})
                risk_model.eval()
                risk_data, risk_dataset, risk_dataloader = None, None, None
            else:
                if len(rb) > 0:
                    for _ in range(args.num_risk_epochs):
                    #if len(rb) > 0:
                        risk_data = rb.sample(args.risk_batch_size)
                        risk_loss = risk_update_step(risk_model, risk_data, risk_criterion, opt_risk, device)
                    logger.store(**{"risk/risk_loss": risk_loss.item()})
                else:
                    logger.store(**{"risk/risk_loss": 0})

        # update policy
        data = buffer.get()
        with torch.no_grad():
            if args.use_risk:
                if epoch < args.start_using_risk:
                    data["risk"] = torch.zeros((data["act"].size()[0], risk_size))
                else:
                    data["risk"] = torch.exp(risk_model(torch.cat([data["obs"], torch.zeros_like(data["act"])], axis=-1))) if args.risk_input == "state_action" else torch.exp(risk_model(data["obs"]))
            else:
                data["risk"] = None 

        # update lagrange multiplier
        ep_costs = logger.get_stats("Metrics/EpCost")
        old_distribution = policy.actor(data["obs"], data["risk"])

        # comnpute advantage
        advantage = data["adv_r"]

        dataloader = DataLoader(
            dataset=TensorDataset(
                data["obs"],
                data["risk"] if args.use_risk else data["obs"],
                data["act"],
                data["log_prob"],
                data["target_value_r"],
                data["target_value_c"],
                advantage,
            ),
            batch_size=config.get("batch_size", args.steps_per_epoch//config.get("num_mini_batch", 1)),
            shuffle=True,
        )
        update_counts = 0
        final_kl = torch.ones_like(old_distribution.loc)
        for _ in range(config["learning_iters"]):
            for (
                obs_b,
                risk_b,
                act_b,
                log_prob_b,
                target_value_r_b,
                target_value_c_b,
                adv_b,
            ) in dataloader:
                risk_b = risk_b if args.use_risk else None
                reward_critic_optimizer.zero_grad()
                loss_r = nn.functional.mse_loss(policy.reward_critic(obs_b, risk_b), target_value_r_b)
                cost_critic_optimizer.zero_grad()
                loss_c = nn.functional.mse_loss(policy.cost_critic(obs_b, risk_b), target_value_c_b)
                if config.get("use_critic_norm", True):
                    for param in policy.reward_critic.parameters():
                        loss_r += param.pow(2).sum() * 0.001
                    for param in policy.cost_critic.parameters():
                        loss_c += param.pow(2).sum() * 0.001
                distribution = policy.actor(obs_b)
                log_prob = distribution.log_prob(act_b).sum(dim=-1)
                ratio = torch.exp(log_prob - log_prob_b)
                ratio_cliped = torch.clamp(ratio, 0.8, 1.2)
                loss_pi = -torch.min(ratio * adv_b, ratio_cliped * adv_b).mean()
                actor_optimizer.zero_grad()
                total_loss = loss_pi + 2*loss_r + loss_c \
                    if config.get("use_value_coefficient", False) \
                    else loss_pi + loss_r + loss_c
                total_loss.backward()
                clip_grad_norm_(policy.parameters(), config["max_grad_norm"])
                reward_critic_optimizer.step()
                cost_critic_optimizer.step()
                actor_optimizer.step()

                logger.store(
                    **{
                        "Loss/Loss_reward_critic": loss_r.mean().item(),
                        "Loss/Loss_cost_critic": loss_c.mean().item(),
                        "Loss/Loss_actor": loss_pi.mean().item(),
                    }
                )

            new_distribution = policy.actor(data["obs"])
            kl = (
                torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
                .sum(-1, keepdim=True)
                .mean()
                .item()
            )
            final_kl = kl
            update_counts += 1
            if kl > config["target_kl"]:
                break
        update_end_time = time.time()
        actor_scheduler.step()
        if not logger.logged:
            # log data
            logger.log_tabular("Metrics/EpRet")
            logger.log_tabular("Metrics/EpCost")
            logger.log_tabular("Metrics/EpLen")
            logger.log_tabular("Metrics/TotalCost")
            logger.log_tabular("Metrics/EpGoal")
            if args.use_eval:
                logger.log_tabular("Metrics/EvalEpRet")
                logger.log_tabular("Metrics/EvalEpCost")
                logger.log_tabular("Metrics/EvalEpLen")
                logger.log_tabular("Metrics/EvalTotalCost")
                logger.log_tabular("Metrics/EvalEpGoal")
            logger.log_tabular("Train/Epoch", epoch + 1)
            logger.log_tabular("Train/TotalSteps", (epoch + 1) * args.steps_per_epoch)
            logger.log_tabular("Train/StopIter", update_counts)
            logger.log_tabular("Train/KL", final_kl)
            logger.log_tabular("Train/LR", actor_scheduler.get_last_lr()[0])
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
            if args.use_risk and args.fine_tune_risk:
                #try:
                logger.log_tabular("risk/risk_loss")
                #except:
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
    wandb.login(key="e5d6d74c569a61c765e1ef12ffffc6d7923ec3db")
    run = wandb.init(config=vars(args), entity="manila95",
                project="risk_aware_exploration",
                monitor_gym=True,
                sync_tensorboard=True, save_code=True)
    relpath = time.strftime("%Y-%m-%d-%H-%M-%S")
    subfolder = "-".join(["seed", str(args.seed).zfill(3)])
    relpath = "-".join([subfolder, relpath])
    algo = os.path.basename(__file__).split(".")[0]
    args.log_dir = os.path.join(args.log_dir, args.experiment if run.sweep_id is None else run.sweep_id, args.task, algo, run.name)
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
