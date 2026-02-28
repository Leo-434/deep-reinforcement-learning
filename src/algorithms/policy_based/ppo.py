import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.algorithms.base import BaseAlgorithm
from src.networks.policies import DiscretePolicy, ContinuousPolicy
from src.networks.mlp import build_mlp

class PPO(BaseAlgorithm):
    """
    Proximal Policy Optimization (PPO).
    Supports both PPO-Clip and PPO-KL depending on the configuration.
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
        
        # PPO specifics
        self.ppo_epochs = cfg["algo"].get("ppo_epochs", 10)
        self.clip_ratio = cfg["algo"].get("clip_ratio", 0.2)
        self.target_kl = cfg["algo"].get("target_kl", 0.01)
        self.use_kl_penalty = cfg["algo"].get("use_kl_penalty", False)
        self.kl_coef = cfg["algo"].get("kl_coef", 0.2)
        self.max_grad_norm = cfg["optim"].get("max_grad_norm", 0.5)
        
        hidden_dims = cfg["network"]["hidden_dims"]
        
        # Policy Network (Actor)
        if self.is_continuous:
            self.actor = ContinuousPolicy(self.obs_dim, self.action_dim, hidden_dims).to(self.device)
        else:
            self.num_actions = cfg["env"]["num_actions"]
            self.actor = DiscretePolicy(self.obs_dim, self.num_actions, hidden_dims).to(self.device)
            
        # Value Network (Critic)
        self.critic = build_mlp(self.obs_dim, 1, hidden_dims).to(self.device)
            
        # Best practice for PPO: Often share optimizers if using shared backbone, 
        # but here we use disjoint networks to keep implementation stable across environments.
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor, eps=1e-5)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic, eps=1e-5)

    def select_action(self, obs: np.ndarray, evaluate: bool = False):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist = self.actor(obs_tensor)
            value = self.critic(obs_tensor)
            if evaluate:
                if self.is_continuous:
                    action = dist.mean
                else:
                    action = torch.argmax(dist.probs, dim=-1)
                log_prob = torch.zeros(1) # Unused in evaluation
            else:
                action = dist.sample()
                log_prob = dist.log_prob(action)
                if self.is_continuous and len(log_prob.shape) > 1:
                    log_prob = log_prob.sum(dim=-1)
                
        # For On-Policy buffer, we need log_probs and values explicitly during stepping
        if self.is_continuous:
            return action.cpu().numpy().flatten(), log_prob.item(), value.item()
        else:
            return action.item(), log_prob.item(), value.item()

    def update(self, batch: dict, steps: int) -> dict:
        obs = batch["obs"]
        acts = batch["acts"]
        returns = batch["returns"]
        advantages = batch["advantages"]
        old_log_probs = batch["log_probs"]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        actor_losses, critic_losses, entropy_losses, kl_divs = [], [], [], []
        
        for _ in range(self.ppo_epochs):
            # Forward passes
            values = self.critic(obs).squeeze(-1)
            dist = self.actor(obs)
            
            if not self.is_continuous:
                acts_ = acts.squeeze(-1).long()
            else:
                acts_ = acts
                
            new_log_probs = dist.log_prob(acts_)
            if self.is_continuous and len(new_log_probs.shape) > 1:
                new_log_probs = new_log_probs.sum(dim=-1)
            entropy = dist.entropy().mean()
            
            # Policy Ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Compute KL Divergence for monitoring / penalty
            with torch.no_grad():
                approx_kl = (old_log_probs - new_log_probs).mean().item()
            
            # Early stopping strictly relying on KL (Best practice in PPO)
            if approx_kl > 1.5 * self.target_kl and not self.use_kl_penalty:
                break
                
            # Actor Loss Strategy
            if self.use_kl_penalty:
                # PPO-KL
                pg_loss = -(ratio * advantages.squeeze(-1)).mean()
                actor_loss = pg_loss + self.kl_coef * approx_kl
            else:
                # PPO-Clip
                surr1 = ratio * advantages.squeeze(-1)
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages.squeeze(-1)
                actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic Loss (MSE)
            # Typically PPO also clips the value function, but unclipped MSE works broadly 
            critic_loss = F.mse_loss(values, returns.squeeze(-1))
            
            # Backprop
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.optimizer_actor.step()
            
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer_critic.step()
            
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropy_losses.append(entropy.item())
            kl_divs.append(approx_kl)
            
        self.learning_steps += 1
        
        return {
            "loss/actor_loss": np.mean(actor_losses) if actor_losses else 0.0,
            "loss/critic_loss": np.mean(critic_losses) if critic_losses else 0.0,
            "loss/entropy": np.mean(entropy_losses) if entropy_losses else 0.0,
            "loss/approx_kl": np.mean(kl_divs) if kl_divs else 0.0
        }

    def save(self, filepath: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer_actor": self.optimizer_actor.state_dict(),
            "optimizer_critic": self.optimizer_critic.state_dict()
        }, filepath)

    def load(self, filepath: str):
        ckpt = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
