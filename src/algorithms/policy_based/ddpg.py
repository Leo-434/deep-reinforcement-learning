import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.algorithms.base import BaseAlgorithm
from src.networks.mlp import build_mlp

class DeterministicPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims, max_action=1.0):
        super().__init__()
        self.net = build_mlp(obs_dim, action_dim, hidden_dims)
        self.max_action = max_action
        
    def forward(self, obs):
        return self.max_action * torch.tanh(self.net(obs))

class DDPG(BaseAlgorithm):
    """
    Deep Deterministic Policy Gradient (DDPG).
    Off-policy, model-free algorithm for continuous action spaces.
    """
    def __init__(self, cfg: dict, device: torch.device):
        super().__init__(cfg, device)
        
        self.obs_dim = cfg["env"]["obs_dim"]
        self.action_dim = cfg["env"]["action_dim"]
        
        # Ensure we are in continuous space
        if not cfg["env"].get("is_continuous", True):
            print("WARNING: DDPG is designed for continuous action spaces.")
            
        self.max_action = cfg["env"].get("max_action", 1.0)
        
        self.gamma = cfg["target"]["gamma"]
        self.tau = cfg["target"]["tau"]
        self.lr_actor = cfg["optim"].get("lr_actor", 1e-4)
        self.lr_critic = cfg["optim"].get("lr_critic", 1e-3)
        self.exploration_noise = cfg["exploration"].get("noise_std", 0.1)
        
        hidden_dims = cfg["network"]["hidden_dims"]
        
        # Actor
        self.actor = DeterministicPolicy(self.obs_dim, self.action_dim, hidden_dims, self.max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        
        # Critic (Q-Network that takes obs + act)
        self.critic = build_mlp(self.obs_dim + self.action_dim, 1, hidden_dims).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
            
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)

    def select_action(self, obs: np.ndarray, evaluate: bool = False) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_tensor).cpu().numpy()[0]
            
        if not evaluate:
            # Gaussian exploration noise
            noise = np.random.normal(0, self.exploration_noise * self.max_action, size=self.action_dim)
            action = np.clip(action + noise, -self.max_action, self.max_action)
            
        return action
        
    def _soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def update(self, batch: dict, steps: int) -> dict:
        obs = batch["obs"]
        acts = batch["acts"]
        rews = batch["rews"]
        next_obs = batch["next_obs"]
        dones = batch["done"]
        
        # --- Update Critic ---
        with torch.no_grad():
            next_actions = self.actor_target(next_obs)
            target_q = self.critic_target(torch.cat([next_obs, next_actions], dim=1))
            target_q = rews + (1 - dones) * self.gamma * target_q
            
        # Get current Q estimate
        current_q = self.critic(torch.cat([obs, acts], dim=1))
        
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()
        
        # --- Update Actor ---
        # Actor seeks to maximize Q value (minimize -Q)
        predicted_actions = self.actor(obs)
        actor_loss = -self.critic(torch.cat([obs, predicted_actions], dim=1)).mean()
        
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
        
        # --- Update Target Networks ---
        self._soft_update(self.critic_target, self.critic, self.tau)
        self._soft_update(self.actor_target, self.actor, self.tau)
        
        self.learning_steps += 1
        
        return {
            "loss/actor_loss": actor_loss.item(), 
            "loss/critic_loss": critic_loss.item()
        }

    def save(self, filepath: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict()
        }, filepath)

    def load(self, filepath: str):
        ckpt = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor"])
        self.critic_target.load_state_dict(ckpt["critic"])
