import numpy as np
import torch
from typing import Dict, Generator

class RolloutBuffer:
    """
    Rollout Buffer for On-Policy algorithms like PPO, A2C, PG.
    Computes Generalized Advantage Estimation (GAE).
    """
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: torch.device, gamma=0.99, gae_lambda=0.95):
        self.capacity = capacity
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.acts = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rews = np.zeros((capacity,), dtype=np.float32)
        self.values = np.zeros((capacity,), dtype=np.float32)
        self.log_probs = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        
        self.advs = np.zeros((capacity,), dtype=np.float32)
        self.returns = np.zeros((capacity,), dtype=np.float32)
        
        self.ptr = 0
        
    def add(self, obs, act, reward, value, log_prob, done):
        if self.ptr >= self.capacity:
            raise IndexError("RolloutBuffer is full")
            
        self.obs[self.ptr] = obs
        self.acts[self.ptr] = act
        self.rews[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        self.ptr += 1
        
    def compute_returns_and_advantage(self, last_value: float, last_done: bool):
        """
        GAE Calculation
        """
        last_gae_lam = 0
        for step in reversed(range(self.ptr)):
            if step == self.ptr - 1:
                next_non_terminal = 1.0 - last_done
                next_values = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]
                
            delta = self.rews[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advs[step] = last_gae_lam
            
        self.returns[:self.ptr] = self.advs[:self.ptr] + self.values[:self.ptr]
        
    def get(self, batch_size: int) -> Generator[Dict[str, torch.Tensor], None, None]:
        indices = np.random.permutation(self.ptr)
        
        for start_idx in range(0, self.ptr, batch_size):
            batch_idxs = indices[start_idx:start_idx + batch_size]
            yield {
                "obs": torch.as_tensor(self.obs[batch_idxs], device=self.device),
                "acts": torch.as_tensor(self.acts[batch_idxs], device=self.device),
                "values": torch.as_tensor(self.values[batch_idxs], device=self.device),
                "log_probs": torch.as_tensor(self.log_probs[batch_idxs], device=self.device),
                "advantages": torch.as_tensor(self.advs[batch_idxs], device=self.device),
                "returns": torch.as_tensor(self.returns[batch_idxs], device=self.device),
            }
            
    def clear(self):
        self.ptr = 0
