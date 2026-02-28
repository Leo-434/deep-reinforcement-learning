import numpy as np
import torch
from typing import Dict

class ReplayBuffer:
    """
    Standard Off-Policy Replay Buffer for algorithms like DQN, DDPG, SAC.
    """
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Memory blocks
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.acts = np.zeros((capacity, action_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.rews = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        
    def add(self, obs: np.ndarray, act: np.ndarray, reward: float, next_obs: np.ndarray, done: bool):
        self.obs[self.ptr] = obs
        self.acts[self.ptr] = act
        self.rews[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        return {
            "obs": torch.as_tensor(self.obs[idxs], dtype=torch.float32, device=self.device),
            "acts": torch.as_tensor(self.acts[idxs], dtype=torch.float32, device=self.device),
            "rews": torch.as_tensor(self.rews[idxs], dtype=torch.float32, device=self.device),
            "next_obs": torch.as_tensor(self.next_obs[idxs], dtype=torch.float32, device=self.device),
            "done": torch.as_tensor(self.done[idxs], dtype=torch.float32, device=self.device)
        }
