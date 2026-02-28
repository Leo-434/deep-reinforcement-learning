import copy
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from src.algorithms.value_based.dqn import DQN
from src.networks.mlp import build_mlp

class QRDQN(DQN):
    """
    Quantile Regression DQN.
    Instead of predicting a single expected Q-value per action, it predicts N quantiles.
    """
    def __init__(self, cfg: dict, device: torch.device):
        super().__init__(cfg, device)
        
        self.num_quantiles = cfg["algo"].get("num_quantiles", 51)
        
        # Override networks to output [Batch, Actions * Quantiles]
        hidden_dims = cfg["network"]["hidden_dims"]
        self.q_net = build_mlp(self.obs_dim, self.num_actions * self.num_quantiles, hidden_dims).to(self.device)
        self.target_net = copy.deepcopy(self.q_net).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        
        # Precompute target quantiles tau_hat
        tau = torch.linspace(0.0, 1.0, self.num_quantiles + 1, device=self.device)
        self.tau_hat = ((tau[:-1] + tau[1:]) / 2.0).view(1, self.num_quantiles)
        
    def get_q_values(self, obs_tensor):
        # Forward pass returning expected Q values
        quantiles = self.q_net(obs_tensor).view(-1, self.num_actions, self.num_quantiles)
        # Expected Q is the mean of quantiles
        q_values = quantiles.mean(dim=2)
        return q_values
        
    def select_action(self, obs: np.ndarray, evaluate: bool = False) -> int:
        if not evaluate and np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
            
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.get_q_values(obs_tensor)
            action = q_values.argmax(dim=-1).item()
            
        return action
        
    def update(self, batch: dict, steps: int) -> dict:
        obs = batch["obs"]
        acts = batch["acts"].long()
        rews = batch["rews"]
        next_obs = batch["next_obs"]
        dones = batch["done"]
        
        batch_size = obs.shape[0]
        
        # Get current quantiles
        # Shape: [Batch, Actions, Quantiles]
        current_quantiles = self.q_net(obs).view(batch_size, self.num_actions, self.num_quantiles)
        # Gather quantiles for selected actions. 
        # acts is [Batch, 1], so we expand it to [Batch, 1, Quantiles] to gather over axis 1
        acts_expanded = acts.unsqueeze(-1).expand(batch_size, 1, self.num_quantiles)
        # Shape: [Batch, Quantiles]
        current_action_quantiles = current_quantiles.gather(1, acts_expanded).squeeze(1)
        
        # Get target quantiles
        with torch.no_grad():
            # Calculate next actions using target net expected Q-values (or online net for Double QRDQN)
            next_q_values = self.get_q_values(next_obs) # [Batch, Actions]
            next_actions = next_q_values.argmax(dim=1, keepdim=True) # [Batch, 1]
            
            # Target quantiles [Batch, Actions, Quantiles] -> gather -> [Batch, Quantiles]
            target_quantiles = self.target_net(next_obs).view(batch_size, self.num_actions, self.num_quantiles)
            next_acts_expanded = next_actions.unsqueeze(-1).expand(batch_size, 1, self.num_quantiles)
            target_action_quantiles = target_quantiles.gather(1, next_acts_expanded).squeeze(1)
            
            # Compute targets using Bellman [Batch, Quantiles]
            target_z = rews + (1 - dones) * self.gamma * target_action_quantiles
            
        # Quantile Huber Loss calculation
        # current_action_quantiles: [Batch, Quantiles_online] -> view [Batch, Quantiles_online, 1]
        # target_z: [Batch, Quantiles_target] -> view [Batch, 1, Quantiles_target]
        td_errors = target_z.unsqueeze(1) - current_action_quantiles.unsqueeze(2)
        
        # Huber Loss
        kappa = 1.0 # Standard for QR-DQN
        huber_loss = F.huber_loss(current_action_quantiles.unsqueeze(2), target_z.unsqueeze(1), reduction='none', delta=kappa)
        
        # Asymmetric weighting
        quantile_weights = torch.abs(self.tau_hat.unsqueeze(-1) - (td_errors.detach() < 0).float())
        quantile_loss = (quantile_weights * huber_loss).sum(dim=2).mean(dim=1).mean()
        
        self.optimizer.zero_grad()
        quantile_loss.backward()
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
        return {"loss/q_loss": quantile_loss.item(), "exploration/epsilon": self.epsilon}
