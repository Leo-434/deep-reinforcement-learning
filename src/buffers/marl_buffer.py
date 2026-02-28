import numpy as np
import torch
from typing import Dict

class MARLReplayBuffer:
    """
    Replay Buffer for Multi-Agent Reinforcement Learning (MARL).
    Stores global states (for CTDE), individual observations, actions, and rewards for multiple agents.
    """
    def __init__(self, capacity: int, num_agents: int, obs_dim: int, state_dim: int, action_dim: int, device: torch.device):
        self.capacity = capacity
        self.num_agents = num_agents
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Dimensions: [capacity, num_agents, dim]
        self.obs = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.acts = np.zeros((capacity, num_agents, action_dim), dtype=np.float32)
        self.rews = np.zeros((capacity, num_agents, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, num_agents, 1), dtype=np.float32)
        
        # Action Masking support [capacity, num_agents, action_dim]
        self.avail_actions = np.zeros((capacity, num_agents, action_dim), dtype=np.float32)
        self.next_avail_actions = np.zeros((capacity, num_agents, action_dim), dtype=np.float32)
        
        # Global states for CTDE
        # Dimension: [capacity, state_dim]
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)

    def add(self, state, obs, act, reward, next_state, next_obs, done, avail_actions=None, next_avail_actions=None):
        self.states[self.ptr] = state
        self.obs[self.ptr] = obs
        self.acts[self.ptr] = act
        self.rews[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done
        
        if avail_actions is not None:
            self.avail_actions[self.ptr] = avail_actions
        else:
            self.avail_actions[self.ptr] = np.ones_like(act)
            
        if next_avail_actions is not None:
            self.next_avail_actions[self.ptr] = next_avail_actions
        else:
            self.next_avail_actions[self.ptr] = np.ones_like(act)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        return {
            "states": torch.as_tensor(self.states[idxs], dtype=torch.float32, device=self.device),
            "obs": torch.as_tensor(self.obs[idxs], dtype=torch.float32, device=self.device),
            "acts": torch.as_tensor(self.acts[idxs], dtype=torch.float32, device=self.device),
            "rews": torch.as_tensor(self.rews[idxs], dtype=torch.float32, device=self.device),
            "next_states": torch.as_tensor(self.next_states[idxs], dtype=torch.float32, device=self.device),
            "next_obs": torch.as_tensor(self.next_obs[idxs], dtype=torch.float32, device=self.device),
            "done": torch.as_tensor(self.done[idxs], dtype=torch.float32, device=self.device),
            "avail_actions": torch.as_tensor(self.avail_actions[idxs], dtype=torch.float32, device=self.device),
            "next_avail_actions": torch.as_tensor(self.next_avail_actions[idxs], dtype=torch.float32, device=self.device),
        }
