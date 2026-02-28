import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.algorithms.policy_based.ddpg import DDPG, DeterministicPolicy
from src.networks.mlp import build_mlp

class TD3(DDPG):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3).
    Addresses overestimation bias in DDPG via Double Critic, Target Smoothing, and Delayed policy updates.
    """
    def __init__(self, cfg: dict, device: torch.device):
        super().__init__(cfg, device)
        
        # Overwrite configs specific to TD3
        self.policy_noise = cfg["exploration"].get("policy_noise", 0.2)
        self.noise_clip = cfg["exploration"].get("noise_clip", 0.5)
        self.policy_freq = cfg["algo"].get("policy_freq", 2)
        
        hidden_dims = cfg["network"]["hidden_dims"]
        
        # We need two Critics for Clipped Double Q-Learning
        self.critic1 = build_mlp(self.obs_dim + self.action_dim, 1, hidden_dims).to(self.device)
        self.critic2 = build_mlp(self.obs_dim + self.action_dim, 1, hidden_dims).to(self.device)
        self.critic1_target = copy.deepcopy(self.critic1).to(self.device)
        self.critic2_target = copy.deepcopy(self.critic2).to(self.device)
        
        # Redefine optimizer for both critics
        self.optimizer_critic = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 
            lr=self.lr_critic
        )

    def update(self, batch: dict, steps: int) -> dict:
        obs = batch["obs"]
        acts = batch["acts"]
        rews = batch["rews"]
        next_obs = batch["next_obs"]
        dones = batch["done"]
        
        # --- Update Critics ---
        with torch.no_grad():
            # Target Policy Smoothing (Adds clipped noise to target actions)
            noise = (torch.randn_like(acts) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_obs) + noise).clamp(-self.max_action, self.max_action)
            
            # Clipped Double Q-Learning
            target_Q1 = self.critic1_target(torch.cat([next_obs, next_actions], dim=1))
            target_Q2 = self.critic2_target(torch.cat([next_obs, next_actions], dim=1))
            target_Q = torch.min(target_Q1, target_Q2)
            
            target_Q = rews + (1 - dones) * self.gamma * target_Q
            
        current_Q1 = self.critic1(torch.cat([obs, acts], dim=1))
        current_Q2 = self.critic2(torch.cat([obs, acts], dim=1))
        
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()
        
        metrics = {"loss/critic_loss": critic_loss.item()}
        
        # --- Delayed Actor Updates ---
        if steps % self.policy_freq == 0:
            predicted_actions = self.actor(obs)
            # Use only critic 1 for policy gradient
            actor_loss = -self.critic1(torch.cat([obs, predicted_actions], dim=1)).mean()
            
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()
            
            # Update targets
            self._soft_update(self.critic1_target, self.critic1, self.tau)
            self._soft_update(self.critic2_target, self.critic2, self.tau)
            self._soft_update(self.actor_target, self.actor, self.tau)
            
            metrics["loss/actor_loss"] = actor_loss.item()
            
        self.learning_steps += 1
        return metrics

    def save(self, filepath: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict()
        }, filepath)

    def load(self, filepath: str):
        ckpt = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.critic1_target.load_state_dict(ckpt["critic1"])
        self.critic2_target.load_state_dict(ckpt["critic2"])
