import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.algorithms.base import BaseAlgorithm
from src.networks.policies import ContinuousPolicy
from src.networks.mlp import build_mlp

class SAC(BaseAlgorithm):
    """
    Soft Actor-Critic (SAC).
    Off-policy maximum entropy Deep RL. Learns to maximize both expected return and policy entropy.
    """
    def __init__(self, cfg: dict, device: torch.device):
        super().__init__(cfg, device)
        
        self.obs_dim = cfg["env"]["obs_dim"]
        self.action_dim = cfg["env"]["action_dim"]
        
        if not cfg["env"].get("is_continuous", True):
            print("WARNING: SAC is designed for continuous action spaces.")
            
        self.gamma = cfg["target"]["gamma"]
        self.tau = cfg["target"]["tau"]
        self.lr_actor = cfg["optim"].get("lr_actor", 3e-4)
        self.lr_critic = cfg["optim"].get("lr_critic", 3e-4)
        self.lr_alpha = cfg["optim"].get("lr_alpha", 3e-4)
        
        # SAC Automatic Entropy Tuning flag
        self.automatic_entropy_tuning = cfg["algo"].get("automatic_entropy_tuning", True)
        self.target_entropy = -torch.prod(torch.Tensor([self.action_dim]).to(self.device)).item()
        
        hidden_dims = cfg["network"]["hidden_dims"]
        
        # Actor
        self.actor = ContinuousPolicy(self.obs_dim, self.action_dim, hidden_dims).to(self.device)
        
        # Twin Critics
        self.critic1 = build_mlp(self.obs_dim + self.action_dim, 1, hidden_dims).to(self.device)
        self.critic2 = build_mlp(self.obs_dim + self.action_dim, 1, hidden_dims).to(self.device)
        self.critic1_target = copy.deepcopy(self.critic1).to(self.device)
        self.critic2_target = copy.deepcopy(self.critic2).to(self.device)
        
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.optimizer_critic = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 
            lr=self.lr_critic
        )
        
        if self.automatic_entropy_tuning:
            # Alpha is the temperature parameter determining the relative importance of the entropy term
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr_alpha)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = cfg["algo"].get("alpha", 0.2)

    def select_action(self, obs: np.ndarray, evaluate: bool = False) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist = self.actor(obs_tensor)
            if evaluate:
                action = dist.mean
            else:
                action = dist.sample()
        return action.cpu().numpy()[0]
        
    def _soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def update(self, batch: dict, steps: int) -> dict:
        obs = batch["obs"]
        acts = batch["acts"]
        rews = batch["rews"]
        next_obs = batch["next_obs"]
        dones = batch["done"]
        
        # --- Update Critics ---
        with torch.no_grad():
            next_dist = self.actor(next_obs)
            next_actions = next_dist.rsample()
            # In continuous spaces, log_prob shape needs sum
            next_log_probs = next_dist.log_prob(next_actions)
            if len(next_log_probs.shape) > 1:
                next_log_probs = next_log_probs.sum(dim=-1).unsqueeze(1)
                
            target_Q1 = self.critic1_target(torch.cat([next_obs, next_actions], dim=1))
            target_Q2 = self.critic2_target(torch.cat([next_obs, next_actions], dim=1))
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_probs
            
            target_Q = rews + (1 - dones) * self.gamma * target_Q
            
        current_Q1 = self.critic1(torch.cat([obs, acts], dim=1))
        current_Q2 = self.critic2(torch.cat([obs, acts], dim=1))
        
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()
        
        # --- Update Actor ---
        dist = self.actor(obs)
        predicted_actions = dist.rsample()
        log_probs = dist.log_prob(predicted_actions)
        if len(log_probs.shape) > 1:
            log_probs = log_probs.sum(dim=-1).unsqueeze(1)
            
        actor_Q1 = self.critic1(torch.cat([obs, predicted_actions], dim=1))
        actor_Q2 = self.critic2(torch.cat([obs, predicted_actions], dim=1))
        actor_Q = torch.min(actor_Q1, actor_Q2)
        
        actor_loss = (self.alpha * log_probs - actor_Q).mean()
        
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
        
        # --- Update Temperature (Alpha) ---
        alpha_loss_val = 0.0
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            
            self.alpha = self.log_alpha.exp().item()
            alpha_loss_val = alpha_loss.item()
            
        # Update targets
        self._soft_update(self.critic1_target, self.critic1, self.tau)
        self._soft_update(self.critic2_target, self.critic2, self.tau)
        
        self.learning_steps += 1
        return {
            "loss/actor_loss": actor_loss.item(), 
            "loss/critic_loss": critic_loss.item(),
            "loss/alpha_loss": alpha_loss_val,
            "metrics/alpha": self.alpha
        }

    def save(self, filepath: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "log_alpha": getattr(self, "log_alpha", None)
        }, filepath)

    def load(self, filepath: str):
        ckpt = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.critic1_target.load_state_dict(ckpt["critic1"])
        self.critic2_target.load_state_dict(ckpt["critic2"])
        if self.automatic_entropy_tuning and ckpt["log_alpha"] is not None:
            self.log_alpha = ckpt["log_alpha"]
            self.alpha = self.log_alpha.exp().item()
