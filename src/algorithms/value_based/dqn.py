import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from src.algorithms.base import BaseAlgorithm
from src.networks.mlp import build_mlp

class DQN(BaseAlgorithm):
    """
    Standard Deep Q-Network.
    """
    def __init__(self, cfg: dict, device: torch.device):
        super().__init__(cfg, device)
        
        self.obs_dim = cfg["env"]["obs_dim"]
        self.num_actions = cfg["env"]["num_actions"]
        
        # Hyperparameters
        self.gamma = cfg["target"]["gamma"]
        self.tau = cfg["target"]["tau"]
        self.lr = cfg["optim"]["lr"]
        
        # Epsilon greedy params
        self.epsilon = cfg["exploration"]["epsilon_start"]
        self.epsilon_min = cfg["exploration"]["epsilon_min"]
        self.epsilon_decay = cfg["exploration"]["epsilon_decay"]
        
        # Networks
        hidden_dims = cfg["network"]["hidden_dims"]
        self.q_net = build_mlp(self.obs_dim, self.num_actions, hidden_dims).to(self.device)
        self.target_net = copy.deepcopy(self.q_net).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)

    def select_action(self, obs: np.ndarray, evaluate: bool = False) -> int:
        # Epsilon-greedy exploration
        if not evaluate and np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
            
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(obs_tensor)
            action = q_values.argmax(dim=-1).item()
            
        return action
        
    def _soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            
    def _hard_update(self, target, source):
        target.load_state_dict(source.state_dict())
        
    def update(self, batch: dict, steps: int) -> dict:
        obs = batch["obs"]
        acts = batch["acts"].long()
        rews = batch["rews"]
        next_obs = batch["next_obs"]
        dones = batch["done"]
        
        # Current Q value
        q_values = self.q_net(obs)
        current_q = q_values.gather(1, acts)
        
        # Target Q value
        with torch.no_grad():
            next_q_values = self.target_net(next_obs)
            max_next_q = next_q_values.max(dim=1, keepdim=True)[0]
            target_q = rews + (1 - dones) * self.gamma * max_next_q
            
        # Loss & Optimize (Both are [batch_size, 1])
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: gradient clipping
        if "max_grad_norm" in self.cfg["optim"]:
            nn.utils.clip_grad_norm_(self.q_net.parameters(), self.cfg["optim"]["max_grad_norm"])
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update target specific type (hard vs soft)
        update_freq = self.cfg["target"].get("update_freq", 1)
        if steps % update_freq == 0:
            if self.tau < 1.0:
                self._soft_update(self.target_net, self.q_net, self.tau)
            else:
                self._hard_update(self.target_net, self.q_net)

        self.learning_steps += 1
        return {"loss/q_loss": loss.item(), "exploration/epsilon": self.epsilon}
        
    def save(self, filepath: str):
        torch.save(self.q_net.state_dict(), filepath)

    def load(self, filepath: str):
        self.q_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
