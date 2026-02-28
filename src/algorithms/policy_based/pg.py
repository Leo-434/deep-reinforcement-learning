import torch
import torch.nn as nn
import numpy as np
from src.algorithms.base import BaseAlgorithm
from src.networks.policies import DiscretePolicy

class PG(BaseAlgorithm):
    """
    Standard Policy Gradient (REINFORCE) Algorithm.
    Supports both discrete (Categorical) and continuous (Gaussian) action spaces.
    """
    def __init__(self, cfg: dict, device: torch.device):
        super().__init__(cfg, device)
        
        self.obs_dim = cfg["env"]["obs_dim"]
        self.action_dim = cfg["env"]["action_dim"]
        self.is_continuous = cfg["env"].get("is_continuous", False)
        
        self.gamma = cfg["target"]["gamma"]
        self.lr = cfg["optim"]["lr"]
        
        hidden_dims = cfg["network"]["hidden_dims"]
        
        # Policy Network
        if self.is_continuous:
            self.policy = GaussianPolicy(self.obs_dim, self.action_dim, hidden_dims).to(self.device)
        else:
            num_actions = cfg["env"]["num_actions"]
            self.policy = DiscretePolicy(self.obs_dim, num_actions, hidden_dims).to(self.device)
            
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def select_action(self, obs: np.ndarray, evaluate: bool = False):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist = self.policy(obs_tensor)
            if evaluate:
                if self.is_continuous:
                    action = dist.mean
                else:
                    action = torch.argmax(dist.probs, dim=-1)
                log_prob = torch.zeros(1)
            else:
                action = dist.sample()
                log_prob = dist.log_prob(action)
                if self.is_continuous and len(log_prob.shape) > 1:
                    log_prob = log_prob.sum(dim=-1)
                
        if self.is_continuous:
            return action.cpu().numpy().flatten(), log_prob.item(), 0.0
        else:
            return action.item(), log_prob.item(), 0.0

    def update(self, batch: dict, steps: int) -> dict:
        obs = batch["obs"]
        acts = batch["acts"]
        
        # PG expects full trajectories usually, RolloutBuffer prepares returns.
        # Here we assume the buffer provides 'returns'
        returns = batch.get("returns", None)
        
        if returns is None:
            raise ValueError("PG requires 'returns' in the batch from RolloutBuffer.")
            
        dist = self.policy(obs)
        
        if not self.is_continuous:
            acts = acts.squeeze(-1).long()
            
        log_probs = dist.log_prob(acts)
        
        # If continuous, dist.log_prob might be [Batch, ActionDim], so we sum over actions
        if self.is_continuous and len(log_probs.shape) > 1:
            log_probs = log_probs.sum(dim=-1)
            
        # Optional: Baseline subtraction for variance reduction
        if "advantages" in batch:
            advantages = batch["advantages"].squeeze(-1)
        else:
            advantages = returns.squeeze(-1)
            
        # REINFORCE Objective: E[ log_prob(a|s) * R ]
        loss = -(log_probs * advantages).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        if "max_grad_norm" in self.cfg["optim"]:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg["optim"]["max_grad_norm"])
        self.optimizer.step()
        
        self.learning_steps += 1
        return {"loss/policy_loss": loss.item(), "metrics/mean_return": returns.mean().item()}

    def save(self, filepath: str):
        torch.save(self.policy.state_dict(), filepath)

    def load(self, filepath: str):
        self.policy.load_state_dict(torch.load(filepath, map_location=self.device))
