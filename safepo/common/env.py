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
try :
    from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.ShadowHandCatchOver2underarm_Safe_finger import ShadowHandCatchOver2Underarm_Safe_finger
    from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.ShadowHandCatchOver2underarm_Safe_joint import ShadowHandCatchOver2Underarm_Safe_joint
    from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.ShadowHandOver_Safe_finger import ShadowHandOver_Safe_finger
    from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.ShadowHandOver_Safe_joint import ShadowHandOver_Safe_joint
    from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.freight_franka_pick_and_place import FreightFrankaPickAndPlace
    from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.freight_franka_close_drawer import FreightFrankaCloseDrawer
    from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.base.multi_vec_task import ShadowHandMultiVecTaskPython, FreightFrankaMultiVecTaskPython
    from safepo.common.wrappers import GymnasiumIsaacEnv
except ImportError:
    pass
from typing import Callable
import safety_gymnasium
from safety_gymnasium.wrappers import SafeAutoResetWrapper, SafeRescaleAction, SafeUnsqueeze
from safety_gymnasium.vector.async_vector_env import SafetyAsyncVectorEnv
from safepo.common.wrappers import ShareSubprocVecEnv, ShareDummyVecEnv, ShareEnv, SafeNormalizeObservation, MultiGoalEnv
import gymnasium as gym
import numpy as np

def make_sa_mujoco_env(num_envs: int, env_id: str, seed: int|None = None, use_aug: bool = False, cost_limit: float = 10.0, horizon: int = 1000):
    """
    Creates and wraps an environment based on the specified parameters.

    Args:
        num_envs (int): Number of parallel environments.
        env_id (str): ID of the environment to create.
        seed (int or None, optional): Seed for the random number generator. Default is None.

    Returns:
        env: The created and wrapped environment.
        obs_space: The observation space of the environment.
        act_space: The action space of the environment.
        
    Examples:
        >>> from safepo.common.env import make_sa_mujoco_env
        >>> 
        >>> env, obs_space, act_space = make_sa_mujoco_env(
        >>>     num_envs=1, 
        >>>     env_id="SafetyPointGoal1-v0", 
        >>>     seed=0
        >>> )
    """
    if num_envs > 1:
        def create_env() -> Callable:
            """Creates an environment that can enable or disable the environment checker."""
            env = safety_gymnasium.make(env_id)
            env = SafeRescaleAction(env, -1.0, 1.0)
            if use_aug:
                print("Using augmented observation")
                env = SafeConstraintBudgetWrapper(env, cost_limit, horizon)
            return env
        env_fns = [create_env for _ in range(num_envs)]
        env = SafetyAsyncVectorEnv(env_fns)
        env = SafeNormalizeObservation(env)
        env.reset(seed=seed)
        obs_space = env.single_observation_space
        act_space = env.single_action_space
    else:
        env = safety_gymnasium.make(env_id)
        if use_aug:
            env = SafeConstraintBudgetWrapper(env, cost_limit, horizon)
        env.reset(seed=seed)
        obs_space = env.observation_space
        act_space = env.action_space
        env = SafeAutoResetWrapper(env)
        env = SafeRescaleAction(env, -1.0, 1.0)
        env = SafeNormalizeObservation(env)
        env = SafeUnsqueeze(env)
    
    return env, obs_space, act_space


class SafeConstraintBudgetWrapper(gym.Wrapper):
    """
    A wrapper that augments the observation with remaining constraint budget and timesteps.
    
    Args:
        env: The environment to wrap
        cost_limit: The maximum allowed cumulative cost per episode
        horizon: The episode horizon/length
    """
    def __init__(self, env, cost_limit, horizon):
        super().__init__(env)
        self.cost_limit = cost_limit
        self.horizon = horizon
        self.current_cost = 0
        self.current_step = 0
        
        # Extend observation space to include budget and time remaining
        if isinstance(env.observation_space, gym.spaces.Dict):
            spaces = env.observation_space.spaces.copy()
            spaces['constraint_budget'] = gym.spaces.Box(
                low=0, high=cost_limit, shape=(1,), dtype=np.float32
            )
            spaces['time_remaining'] = gym.spaces.Box(
                low=0, high=horizon, shape=(1,), dtype=np.float32
            )
            self.observation_space = gym.spaces.Dict(spaces)
        else:
            low = np.concatenate([env.observation_space.low, [0, 0]])
            high = np.concatenate([env.observation_space.high, [cost_limit, horizon]])
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_cost = 0
        self.current_step = 0
        return self._augment_observation(obs), info

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        self.current_cost += cost
        self.current_step += 1
        return self._augment_observation(obs), reward, cost, terminated, truncated, info
        
    def _augment_observation(self, obs):
        remaining_budget = self.cost_limit - self.current_cost
        remaining_time = self.horizon - self.current_step
        
        if isinstance(obs, dict):
            obs = obs.copy()
            obs['constraint_budget'] = np.array([remaining_budget], dtype=np.float32)
            obs['time_remaining'] = np.array([remaining_time], dtype=np.float32)
            return obs
        else:
            return np.concatenate([
                obs,
                np.array([remaining_budget, remaining_time], dtype=np.float32)
            ])




def make_sa_isaac_env(args, cfg, sim_params):
    """
    Creates and returns a VecTaskPython environment for the single agent Isaac Gym task.

    Args:
        args: Command-line arguments.
        cfg: Configuration for the environment.
        cfg_train: Training configuration.
        sim_params: Parameters for the simulation.

    Returns:
        env: VecTaskPython environment for the single agent Isaac Gym task.

    Warning:
        SafePO's single agent Isaac Gym task is not ready for use yet.
    """
    # create native task and pass custom config
    device_id = args.device_id
    rl_device = args.device

    cfg["seed"] = args.seed
    cfg_task = cfg["env"]
    cfg_task["seed"] = cfg["seed"]
    task = eval(args.task)(
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        device_type=args.device,
        device_id=device_id,
        headless=args.headless,
        is_multi_agent=False)
    try:
        env = GymnasiumIsaacEnv(task, rl_device)
    except ModuleNotFoundError:
        env = None

    return env

def make_ma_mujoco_env(scenario, agent_conf, seed, cfg_train):
    """
    Creates and returns a multi-agent environment using MuJoCo scenarios.

    Args:
        args: Command-line arguments.
        cfg_train: Training configuration.

    Returns:
        env: A multi-agent environment.
    """
    def get_env_fn(rank):
        def init_env():
            """
            Initializes and returns a ShareEnv instance for the given rank.

            Returns:
                env: Initialized ShareEnv instance.
            """
            env=ShareEnv(
                scenario=scenario,
                agent_conf=agent_conf,
            )
            env.reset(seed=seed + rank * 1000)
            return env

        return init_env

    if cfg_train['n_rollout_threads']== 1:
        return ShareDummyVecEnv([get_env_fn(0)], cfg_train['device'])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(cfg_train['n_rollout_threads'])])

def make_ma_multi_goal_env(task, seed, cfg_train):
    """
    Creates and returns a multi-agent environment using MuJoCo scenarios.

    Args:
        args: Command-line arguments.
        cfg_train: Training configuration.

    Returns:
        env: A multi-agent environment.
    """
    def get_env_fn(rank):
        def init_env():
            """
            Initializes and returns a ShareEnv instance for the given rank.

            Returns:
                env: Initialized ShareEnv instance.
            """
            env=MultiGoalEnv(
                task=task,
                seed=seed,
            )
            return env

        return init_env
    
    if cfg_train['n_rollout_threads']== 1:
        return ShareDummyVecEnv([get_env_fn(0)], cfg_train['device'])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(cfg_train['n_rollout_threads'])])

def make_ma_isaac_env(args, cfg, cfg_train, sim_params, agent_index):
    """
    Creates and returns a multi-agent environment for the Isaac Gym task.

    Args:
        args: Command-line arguments.
        cfg: Configuration for the environment.
        cfg_train: Training configuration.
        sim_params: Parameters for the simulation.
        agent_index: Index of the agent within the multi-agent environment.

    Returns:
        env: A multi-agent environment for the Isaac Gym task.
    """
    # create native task and pass custom config
    device_id = args.device_id
    rl_device = args.device

    cfg["seed"] = cfg_train.get("seed", -1)
    cfg_task = cfg["env"]
    cfg_task["seed"] = cfg["seed"]
    task = eval(args.task)(
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        device_type=args.device,
        device_id=device_id,
        headless=args.headless,
        agent_index=agent_index,
        is_multi_agent=True)
    task_name = task.__class__.__name__
    if "ShadowHand" in task_name:
        env = ShadowHandMultiVecTaskPython(task, rl_device)
    elif "FreightFranka" in task_name:
        env = FreightFrankaMultiVecTaskPython(task, rl_device)
    else:
        raise NotImplementedError

    return env
