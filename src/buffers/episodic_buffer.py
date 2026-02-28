import numpy as np
import torch
from typing import Dict, List

class EpisodicReplayBuffer:
    """
    Episodic Replay Buffer for Recurrent algorithms like DRQN.
    Stores full episodes instead of step transitions to allow BPTT (Backprop Through Time).
    """
    def __init__(self, capacity_episodes: int, max_episode_len: int, obs_dim: int, action_dim: int, device: torch.device):
        self.capacity = capacity_episodes
        self.max_episode_len = max_episode_len
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # [num_episodes, max_ep_len, dim]
        self.obs = np.zeros((self.capacity, max_episode_len, obs_dim), dtype=np.float32)
        self.acts = np.zeros((self.capacity, max_episode_len, action_dim), dtype=np.float32)
        self.rews = np.zeros((self.capacity, max_episode_len, 1), dtype=np.float32)
        # We need an explicit unpadded mask to handle variable episode lengths
        self.mask = np.zeros((self.capacity, max_episode_len, 1), dtype=np.float32)
        
        self.current_ep_obs = []
        self.current_ep_acts = []
        self.current_ep_rews = []
        
    def add_step(self, obs, act, reward, done):
        self.current_ep_obs.append(obs)
        self.current_ep_acts.append(act)
        self.current_ep_rews.append(reward)
        
        # When episode finishes or max len reached
        if done or len(self.current_ep_obs) >= self.max_episode_len:
            self.flush_episode()

    def flush_episode(self):
        ep_len = len(self.current_ep_obs)
        if ep_len == 0: return

        # Padding
        padded_obs = np.zeros((self.max_episode_len, self.obs.shape[-1]), dtype=np.float32)
        padded_acts = np.zeros((self.max_episode_len, self.acts.shape[-1]), dtype=np.float32)
        padded_rews = np.zeros((self.max_episode_len, 1), dtype=np.float32)
        mask = np.zeros((self.max_episode_len, 1), dtype=np.float32)
        
        padded_obs[:ep_len] = np.array(self.current_ep_obs)
        padded_acts[:ep_len] = np.array(self.current_ep_acts).reshape(ep_len, -1)
        padded_rews[:ep_len] = np.array(self.current_ep_rews).reshape(-1, 1)
        mask[:ep_len] = 1.0
        
        self.obs[self.ptr] = padded_obs
        self.acts[self.ptr] = padded_acts
        self.rews[self.ptr] = padded_rews
        self.mask[self.ptr] = mask
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
        # Clear buffer
        self.current_ep_obs.clear()
        self.current_ep_acts.clear()
        self.current_ep_rews.clear()

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        return {
            "obs": torch.as_tensor(self.obs[idxs], dtype=torch.float32, device=self.device),
            "acts": torch.as_tensor(self.acts[idxs], dtype=torch.float32, device=self.device),
            "rews": torch.as_tensor(self.rews[idxs], dtype=torch.float32, device=self.device),
            "mask": torch.as_tensor(self.mask[idxs], dtype=torch.float32, device=self.device)
        }
