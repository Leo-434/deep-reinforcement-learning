import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.algorithms.base import BaseAlgorithm
from src.networks.policies import DiscretePolicy, ContinuousPolicy
from src.networks.mlp import build_mlp

class A2C(BaseAlgorithm):
    """
    Advantage Actor Critic (A2C).
    Learns both a Policy (Actor) and a Value function (Critic) to reduce variance via Advantage estimation.
    """
    def __init__(self, cfg: dict, device: torch.device):
        super().__init__(cfg, device)
        
        self.obs_dim = cfg["env"]["obs_dim"]
        self.action_dim = cfg["env"]["action_dim"]
        self.is_continuous = cfg["env"].get("is_continuous", False)
        
        self.gamma = cfg["target"]["gamma"]
        self.lr_actor = cfg["optim"].get("lr_actor", 3e-4)
        self.lr_critic = cfg["optim"].get("lr_critic", 1e-3)
        self.entropy_coef = cfg["algo"].get("entropy_coef", 0.01)
        self.value_coef = cfg["algo"].get("value_coef", 0.5)
        
        hidden_dims = cfg["network"]["hidden_dims"]
        
        # Policy Network (Actor)
        if self.is_continuous:
            self.actor = ContinuousPolicy(self.obs_dim, self.action_dim, hidden_dims).to(self.device)
        else:
            self.num_actions = cfg["env"]["num_actions"]
            self.actor = DiscretePolicy(self.obs_dim, self.num_actions, hidden_dims).to(self.device)
            
        # Value Network (Critic)
        self.critic = build_mlp(self.obs_dim, 1, hidden_dims).to(self.device)
            
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)

    def select_action(self, obs: np.ndarray, evaluate: bool = False):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist = self.actor(obs_tensor)
            if evaluate:
                if self.is_continuous:
                    action = dist.mean
                else:
                    action = torch.argmax(dist.probs, dim=-1)
                    
                if self.is_continuous:
                    return action.cpu().numpy()[0]
                else:
                    return action.item()
            else:
                action = dist.sample()
                log_prob = dist.log_prob(action)
                if self.is_continuous and len(log_prob.shape) > 1:
                    log_prob = log_prob.sum(dim=-1)
                value = self.critic(obs_tensor)
                
                if self.is_continuous:
                    return action.cpu().numpy()[0], log_prob.item(), value.item()
                else:
                    return action.item(), log_prob.item(), value.item()
            
    def get_value(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            value = self.critic(obs_tensor)
        return value.item()

    def update(self, batch: dict, steps: int) -> dict:
        obs = batch["obs"]
        acts = batch["acts"]
        returns = batch["returns"]
        advantages = batch["advantages"]
        
        # Forward pass Critic
        values = self.critic(obs)
        
        # Forward pass Actor
        dist = self.actor(obs)
        if not self.is_continuous:
            acts = acts.squeeze(-1).long()
            
        log_probs = dist.log_prob(acts)
        entropy = dist.entropy()
        
        if self.is_continuous and len(log_probs.shape) > 1:
            log_probs = log_probs.sum(dim=-1)
            entropy = entropy.sum(dim=-1)
            
        # Actor Loss
        actor_loss = -(log_probs * advantages.squeeze(-1)).mean()
        
        # Critic Loss (MSE between Value and Returns)
        critic_loss = F.mse_loss(values.squeeze(-1), returns)
        
        # Total Loss
        entropy_loss = entropy.mean()
        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy_loss
        
        # Optimize iteratively or together
        # We'll optimize together for simplicity if networks share trunk, but here they are separate.
        # Doing separate step passes is cleaner for unshared networks.
        
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        
        if "max_grad_norm" in self.cfg["optim"]:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg["optim"]["max_grad_norm"])
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg["optim"]["max_grad_norm"])
            
        self.optimizer_actor.step()
        self.optimizer_critic.step()
        
        self.learning_steps += 1
        
        return {
            "loss/actor_loss": actor_loss.item(), 
            "loss/critic_loss": critic_loss.item(),
            "loss/entropy": entropy_loss.item()
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
