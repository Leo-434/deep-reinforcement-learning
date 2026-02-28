import os
import argparse
import torch
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.logger import Logger
from src.envs.make_env import make_env
from src.buffers.replay_buffer import ReplayBuffer
from src.algorithms.value_based.dqn import DQN
from src.algorithms.value_based.ddqn import DDQN
from src.algorithms.value_based.dueling_dqn import DuelingDQN
from src.algorithms.value_based.per_dqn import PERDQN
from src.algorithms.value_based.noisy_dqn import NoisyDQN
from src.algorithms.value_based.drqn import DRQN
from src.algorithms.value_based.qr_dqn import QRDQN
from src.algorithms.value_based.c51 import C51

# Policy-based imports
from src.algorithms.policy_based.pg import PG
from src.algorithms.policy_based.a2c import A2C
from src.algorithms.policy_based.ppo import PPO
from src.algorithms.policy_based.ddpg import DDPG
from src.algorithms.policy_based.td3 import TD3
from src.algorithms.policy_based.sac import SAC

# Multi-Agent imports
from src.algorithms.multi_agent.value_based.iql import IQL
from src.algorithms.multi_agent.value_based.qmix import QMIX
from src.algorithms.multi_agent.policy_based.ippo import IPPO
from src.algorithms.multi_agent.policy_based.mappo import MAPPO
from src.algorithms.multi_agent.policy_based.maddpg import MADDPG

from src.buffers.per_buffer import PrioritizedReplayBuffer
from src.buffers.episodic_buffer import EpisodicReplayBuffer
from src.buffers.marl_buffer import MARLReplayBuffer

# Dynamic registry mapping names to models
ALGO_REGISTRY = {
    # Single Agent Value Based
    "DQN": DQN,
    "DDQN": DDQN,
    "DuelingDQN": DuelingDQN,
    "PERDQN": PERDQN,
    "NoisyDQN": NoisyDQN,
    "DRQN": DRQN,
    "QRDQN": QRDQN,
    "C51": C51,
    
    # Single Agent Policy Based
    "PG": PG,
    "A2C": A2C,
    "PPO": PPO,
    "DDPG": DDPG,
    "TD3": TD3,
    "SAC": SAC,
    
    # Multi-Agent Algorithms
    "IQL": IQL,
    "QMIX": QMIX,
    "IPPO": IPPO,
    "MAPPO": MAPPO,
    "MADDPG": MADDPG
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device()
    print(f"Using device: {device}")
    
    # Setup Logger
    logger = Logger(log_dir=cfg["env"]["log_dir"])
    
    is_marl = cfg["env"].get("is_marl", False)
    
    # 1. Provide Environment
    if is_marl:
        from src.envs.make_marl import make_marl_env
        env = make_marl_env(cfg["env"]["env_id"])
        env.reset(seed=cfg["env"].get("seed", 42))
        obs, info = env.reset()
        # Extract dimensions from PettingZoo ParallelEnv
        agents = env.agents
        cfg["env"]["num_agents"] = len(agents)
        cfg["env"]["obs_dim"] = env.observation_space(agents[0]).shape[0]
        
        import gymnasium as gym
        action_space = env.action_space(agents[0])
        if isinstance(action_space, gym.spaces.Discrete):
            cfg["env"]["action_dim"] = 1
            cfg["env"]["num_actions"] = action_space.n
            cfg["env"]["is_continuous"] = False
        else:
            cfg["env"]["action_dim"] = action_space.shape[0]
            cfg["env"]["is_continuous"] = True
            
        # Optional global state for CTDE
        try:
            cfg["env"]["state_dim"] = env.state_space.shape[0]
        except (AttributeError, NotImplementedError):
            # Fallback: concatenate all obs
            cfg["env"]["state_dim"] = cfg["env"]["obs_dim"] * cfg["env"]["num_agents"]
            
    else:
        env = make_env(cfg["env"]["env_id"], seed=cfg["env"].get("seed", 42))
        # Automatically infer dimensions and inject into config
        import gymnasium as gym
        cfg["env"]["obs_dim"] = env.observation_space.shape[0]
        if isinstance(env.action_space, gym.spaces.Discrete):
            cfg["env"]["action_dim"] = 1
            cfg["env"]["num_actions"] = env.action_space.n # Required by policy networks
            cfg["env"]["is_continuous"] = False
        else:
            cfg["env"]["action_dim"] = env.action_space.shape[0]
            cfg["env"]["is_continuous"] = True
            
        obs, info = env.reset(seed=cfg["env"].get("seed", 42))
    
    # 2. Provide Buffer
    buffer_cap = cfg["buffer"]["capacity"]
    algo_name = cfg["algo"]["name"]
    
    if is_marl:
        buffer = MARLReplayBuffer(buffer_cap, cfg["env"]["num_agents"], cfg["env"]["obs_dim"], cfg["env"]["state_dim"], cfg["env"]["action_dim"], device)
    elif algo_name in ["PPO", "A2C", "PG", "NPG", "PPG"]:
        from src.buffers.rollout_buffer import RolloutBuffer
        buffer = RolloutBuffer(buffer_cap, cfg["env"]["obs_dim"], cfg["env"]["action_dim"], device, 
                               gamma=cfg["target"].get("gamma", 0.99), 
                               gae_lambda=cfg["target"].get("gae_lambda", 0.95))
    elif algo_name == "PERDQN":
        buffer = PrioritizedReplayBuffer(buffer_cap, cfg["env"]["obs_dim"], cfg["env"]["action_dim"], device)
    elif algo_name == "DRQN":
        max_ep_len = cfg["buffer"].get("max_episode_len", 500)
        buffer = EpisodicReplayBuffer(buffer_cap, max_ep_len, cfg["env"]["obs_dim"], cfg["env"]["action_dim"], device)
    else:
        buffer = ReplayBuffer(buffer_cap, cfg["env"]["obs_dim"], cfg["env"]["action_dim"], device)
    
    # 3. Provide Algorithm
    algo_class = ALGO_REGISTRY.get(algo_name)
    if not algo_class:
        raise ValueError(f"Algorithm {algo_name} not found in ALGO_REGISTRY")
    agent = algo_class(cfg, device)
    
    # 4. Training Loop
    max_steps = cfg["training"].get("max_steps", 100000)
    batch_size = cfg["training"].get("batch_size", 32)
    learning_starts = cfg["training"].get("learning_starts", 1000)
    train_freq = cfg["training"].get("train_freq", 1)
    
    episode_reward = 0
    episodes_completed = 0
    
    # Pre-compute policy type since it's static
    is_on_policy = hasattr(buffer, "compute_returns_and_advantage")
    
    try:
        for step in range(1, max_steps + 1):
            if is_marl:
                # MARL Parallel Loop
                import numpy as np
                
                # Extract obs dict to array: [num_agents, obs_dim]
                obs_array = np.array([obs[a] for a in env.agents])
                
                # Build avail_actions if supported by env, otherwise None
                avail_actions = None
                if "action_mask" in info.get(env.agents[0], {}):
                    avail_actions = np.array([info[a]["action_mask"] for a in env.agents])
                
                if step < learning_starts:
                    action_dict = {a: env.action_space(a).sample() for a in env.agents}
                    actions_array = np.array([[action_dict[a]] for a in env.agents])
                else:
                    actions_array = agent.select_action(obs_array, evaluate=False, avail_actions=avail_actions)
                    # Ensure actions_array is [N, ActionDim]
                    if len(actions_array.shape) == 1:
                        actions_array = np.expand_dims(actions_array, axis=-1)
                    action_dict = {a: actions_array[i, 0] if cfg["env"].get("num_actions") else actions_array[i] for i, a in enumerate(env.agents)}
                    
                next_obs, rewards, terminations, truncations, infos = env.step(action_dict)
                
                # Aggregate returns (assumes cooperative shared reward for now)
                reward = list(rewards.values())[0] if rewards else 0.0
                episode_reward += reward
                
                done = any(terminations.values()) or any(truncations.values())
                
                # PettingZoo removes agents from dicts when they are done. 
                # We must fallback to their last obs to keep array dimensions valid for the buffer.
                next_obs_array = np.array([next_obs.get(a, obs[a]) for a in env.possible_agents])
                next_avail_actions = None
                
                # Attempt to gather next action masks, fallback to current if missing
                masks = []
                has_masks = False
                for a in env.possible_agents:
                    m = infos.get(a, info.get(a, {})).get("action_mask", None)
                    if m is not None:
                        has_masks = True
                    masks.append(m)
                    
                if has_masks:
                    # Fill any None masks with ones
                    for i in range(len(masks)):
                        if masks[i] is None:
                            masks[i] = np.ones(cfg["env"]["action_dim"])
                    next_avail_actions = np.array(masks)
                
                # Centralized States
                try:
                    state_array = env.state()
                except (AttributeError, NotImplementedError):
                    state_array = obs_array.flatten()
                    
                try:
                    # Approximate next state as state isn't explicitly returned in step
                    next_state_array = env.state()
                except (AttributeError, NotImplementedError):
                    next_state_array = next_obs_array.flatten()
                    
                # Add to buffer, using strict termination for bootstrapping
                buffer_term = any(terminations.values())
                buffer.add(
                    state_array, 
                    obs_array, 
                    actions_array, 
                    np.ones((cfg["env"]["num_agents"], 1)) * reward, 
                    next_state_array, 
                    next_obs_array, 
                    np.ones((cfg["env"]["num_agents"], 1)) * int(buffer_term),
                    avail_actions=avail_actions,
                    next_avail_actions=next_avail_actions
                )
                
                obs = next_obs
                info = infos
                
            else:
                # Single Agent Loop
                log_prob, value = None, None
                
                if step < learning_starts and not is_on_policy:
                    action = env.action_space.sample()
                else:
                    if is_on_policy:
                        action, log_prob, value = agent.select_action(obs, evaluate=False)
                    else:
                        action = agent.select_action(obs, evaluate=False)
                        
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                # Use 'terminated' to mask future Q-values, properly bootstrapping 'truncated' states
                buffer_done = terminated

                # Store in buffer
                if is_on_policy:
                    buffer.add(obs, action, reward, value, log_prob, buffer_done)
                elif getattr(buffer, "add_step", None):
                    buffer.add_step(obs, action, reward, done) # Episodic buffer still needs episode boundaries
                else:
                    buffer.add(obs, action, reward, next_obs, buffer_done)
                    
                obs = next_obs
                
            if done:
                logger.log_episode_reward(episode_reward, step)
                episodes_completed += 1
                obs, info = env.reset()
                if getattr(agent, "reset_hidden", None):
                    agent.reset_hidden()
                    
                episode_reward = 0
                
            # Update Agent
            if step >= learning_starts and step % train_freq == 0:
                if is_on_policy:
                    with torch.no_grad():
                        if hasattr(agent, "critic"):
                            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                            last_value = agent.critic(obs_tensor).item()
                        else:
                            last_value = 0.0
                    buffer.compute_returns_and_advantage(last_value, done)
                    
                    # PPO handles epochs internally over the provided batch
                    for batch in buffer.get(buffer.capacity):
                        metrics = agent.update(batch, step)
                        break 
                    buffer.clear()
                else:
                    if getattr(buffer, "sample", None):
                        batch = buffer.sample(batch_size)
                        metrics = agent.update(batch, step)
                    
                        # Special case for PER buffer priority updates
                        if "_priorities" in metrics:
                            tree_idxs, abs_errors = metrics.pop("_priorities")
                            buffer.update_priorities(tree_idxs, abs_errors)
                
                # Log metrics loosely every 100 updates max to avoid IO bottleneck
                if step % (train_freq * 100) == 0:
                    for k, v in metrics.items():
                        logger.log_scalar(k, v, step)
                        
            # Optional: Save every N steps
            save_freq = cfg["training"].get("save_freq", 10000)
            if step % save_freq == 0:
                os.makedirs(cfg["env"]["save_dir"], exist_ok=True)
                agent.save(os.path.join(cfg["env"]["save_dir"], f"{algo_name}_{step}.pth"))

    except KeyboardInterrupt:
        print("\nTraining gracefully interrupted by User.")
        
    finally:
        # Final shutdown guarantees
        os.makedirs(cfg["env"]["save_dir"], exist_ok=True)
        logger.save_graphics()
        agent.save(os.path.join(cfg["env"]["save_dir"], f"{algo_name}_final.pth"))
        env.close()
        logger.close()
        print("Training Diagnostics Saved Successfully!")

if __name__ == "__main__":
    main()
