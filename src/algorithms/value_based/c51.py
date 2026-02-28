import copy
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from src.algorithms.value_based.dqn import DQN
from src.networks.mlp import build_mlp

class C51(DQN):
    """
    Categorical 51 DQN (C51).
    Learns a categorical distribution over a fixed set of returns (supports) instead of expected values.
    """
    def __init__(self, cfg: dict, device: torch.device):
        super().__init__(cfg, device)
        
        self.num_atoms = cfg["algo"].get("num_atoms", 51)
        self.v_min = cfg["algo"].get("v_min", -10.0)
        self.v_max = cfg["algo"].get("v_max", 10.0)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        
        # Support vector [num_atoms]
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(self.device)
        
        # Override networks to output logits for [Actions, Atoms]
        hidden_dims = cfg["network"]["hidden_dims"]
        self.q_net = build_mlp(self.obs_dim, self.num_actions * self.num_atoms, hidden_dims).to(self.device)
        self.target_net = copy.deepcopy(self.q_net).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        
    def get_action_probs(self, obs_tensor, net):
        # Forward pass returning softmax probabilities
        batch_size = obs_tensor.shape[0]
        logits = net(obs_tensor).view(batch_size, self.num_actions, self.num_atoms)
        probs = F.softmax(logits, dim=-1) # [Batch, Actions, Atoms]
        return probs
        
    def get_q_values(self, obs_tensor, net):
        probs = self.get_action_probs(obs_tensor, net)
        # Expected value is sum of (prob * support)
        q_values = (probs * self.support.view(1, 1, self.num_atoms)).sum(dim=-1)
        return q_values
        
    def select_action(self, obs: np.ndarray, evaluate: bool = False) -> int:
        if not evaluate and np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
            
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.get_q_values(obs_tensor, self.q_net)
            action = q_values.argmax(dim=-1).item()
            
        return action
        
    def update(self, batch: dict, steps: int) -> dict:
        obs = batch["obs"]
        acts = batch["acts"].long()
        rews = batch["rews"]
        next_obs = batch["next_obs"]
        dones = batch["done"]
        
        batch_size = obs.shape[0]
        
        # Current Logits and Probs
        logits = self.q_net(obs).view(batch_size, self.num_actions, self.num_atoms)
        log_probs = F.log_softmax(logits, dim=-1) # Log-probs for Cross Entropy
        
        # Gather log-probs for selected actions [Batch, Atoms]
        acts_expanded = acts.unsqueeze(-1).expand(batch_size, 1, self.num_atoms)
        current_action_log_probs = log_probs.gather(1, acts_expanded).squeeze(1)
        
        # Project target distribution
        with torch.no_grad():
            # Select max action using target net
            next_q_values = self.get_q_values(next_obs, self.target_net)
            next_actions = next_q_values.argmax(dim=1, keepdim=True)
            
            # Next state probability distribution [Batch, Atoms]
            next_probs = self.get_action_probs(next_obs, self.target_net)
            next_acts_expanded = next_actions.unsqueeze(-1).expand(batch_size, 1, self.num_atoms)
            next_action_probs = next_probs.gather(1, next_acts_expanded).squeeze(1)
            
            # Compute projected supports
            # tz: [Batch, Atoms]
            tz = rews + (1 - dones) * self.gamma * self.support.view(1, self.num_atoms)
            tz = tz.clamp(self.v_min, self.v_max)
            
            # Distribute probabilities into target distribution explicitly
            b = (tz - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            
            # Fix identical indices
            l[(u > 0) & (l == u)] -= 1
            u[(l < (self.num_atoms - 1)) & (l == u)] += 1
            
            # Project probabilities
            target_probs = torch.zeros_like(next_action_probs)
            offset = torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size, device=self.device).long().unsqueeze(1)
            
            target_probs.view(-1).index_add_(0, (l + offset).view(-1), (next_action_probs * (u.float() - b)).view(-1))
            target_probs.view(-1).index_add_(0, (u + offset).view(-1), (next_action_probs * (b - l.float())).view(-1))
            
        # Cross-Entropy Loss
        loss = -(target_probs * current_action_log_probs).sum(dim=-1).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        if "max_grad_norm" in self.cfg["optim"]:
            nn.utils.clip_grad_norm_(self.q_net.parameters(), self.cfg["optim"]["max_grad_norm"])
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        update_freq = self.cfg["target"].get("update_freq", 1)
        if steps % update_freq == 0:
            if self.tau < 1.0:
                self._soft_update(self.target_net, self.q_net, self.tau)
            else:
                self._hard_update(self.target_net, self.q_net)

        self.learning_steps += 1
        return {"loss/q_loss": loss.item(), "exploration/epsilon": self.epsilon}
