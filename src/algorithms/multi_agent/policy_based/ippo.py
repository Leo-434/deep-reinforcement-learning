import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.algorithms.base import BaseAlgorithm
from src.networks.policies import DiscretePolicy
from src.networks.mlp import build_mlp

class IPPO(BaseAlgorithm):
    """
    Independent Proximal Policy Optimization (IPPO).
    Treats each agent independently utilizing the shared networks for homogeneous policies.
    """
    def __init__(self, cfg: dict, device: torch.device):
        super().__init__(cfg, device)
        
        self.num_agents = cfg["env"]["num_agents"]
        self.obs_dim = cfg["env"]["obs_dim"]
        self.num_actions = cfg["env"]["num_actions"]
        
        self.gamma = cfg["target"]["gamma"]
        self.lam = cfg["target"].get("lam", 0.95)
        
        self.clip_ratio = cfg["algo"].get("clip_ratio", 0.2)
        self.ppo_epochs = cfg["algo"].get("ppo_epochs", 4)
        self.entropy_coef = cfg["algo"].get("entropy_coef", 0.01)
        self.vf_coef = cfg["algo"].get("vf_coef", 0.5)
        
        self.lr_actor = cfg["optim"].get("lr_actor", 3e-4)
        self.lr_critic = cfg["optim"].get("lr_critic", 1e-3)
        
        hidden_dims = cfg["network"]["hidden_dims"]
        
        # Shared Actor (Input: Local Obs)
        self.actor = DiscretePolicy(self.obs_dim, self.num_actions, hidden_dims).to(self.device)
        
        # Shared Critic (Input: Local Obs) -> IQL style
        self.critic = build_mlp(self.obs_dim, 1, hidden_dims).to(self.device)
        
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)

    def select_action(self, obs: np.ndarray, evaluate: bool = False, avail_actions: np.ndarray = None) -> np.ndarray:
        actions = []
        for agent_id in range(self.num_agents):
            agent_obs = obs[agent_id]
            obs_tensor = torch.as_tensor(agent_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            with torch.no_grad():
                dist = self.actor(obs_tensor)
                
                # Manual Masking for distributions
                if avail_actions is not None:
                    agent_avail = avail_actions[agent_id]
                    mask = torch.as_tensor(agent_avail, device=self.device).bool()
                    dist.logits[~mask] = -1e10
                
                if evaluate:
                    action = dist.probs.argmax(dim=-1).item()
                else:
                    action = dist.sample().item()
            actions.append(action)
            
        return np.array(actions)
        
    def compute_gae(self, rews, vals, next_vals, dones):
        advs = torch.zeros_like(rews).to(self.device)
        last_gae_lam = 0
        for t in reversed(range(rews.size(0))):
            if t == rews.size(0) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_val = next_vals[t]
            else:
                next_non_terminal = 1.0 - dones[t]
                next_val = vals[t+1]
            
            delta = rews[t] + self.gamma * next_val * next_non_terminal - vals[t]
            advs[t] = last_gae_lam = delta + self.gamma * self.lam * next_non_terminal * last_gae_lam
        
        rets = advs + vals
        return advs, rets

    def update(self, batch: dict, steps: int) -> dict:
        obs = batch["obs"]              # [Seq, N_Agents, ObsDim]
        acts = batch["acts"].long()     # [Seq, N_Agents, 1]
        rews = batch["rews"]            # [Seq, N_Agents, 1]
        next_obs = batch["next_obs"]    # [Seq, N_Agents, ObsDim]
        dones = batch["done"]           # [Seq, N_Agents, 1]
        avail_actions = batch.get("avail_actions", None) # [Seq, N_Agents, NumActions]
        
        seq_len = obs.shape[0]
        
        # Merge batch and agent dimensions
        obs_flat = obs.view(seq_len * self.num_agents, self.obs_dim)
        acts_flat = acts.view(seq_len * self.num_agents, 1)
        rews_flat = rews.view(seq_len * self.num_agents, 1)
        dones_flat = dones.view(seq_len * self.num_agents, 1)
        next_obs_flat = next_obs.view(seq_len * self.num_agents, self.obs_dim)
        if avail_actions is not None:
            avail_flat = avail_actions.view(seq_len * self.num_agents, self.num_actions)
        else:
            avail_flat = None
        
        with torch.no_grad():
            old_vals = self.critic(obs_flat).view(seq_len, self.num_agents, 1)
            next_vals = self.critic(next_obs_flat).view(seq_len, self.num_agents, 1)
            
            # Reconstruct sequence to compute proper GAE across time steps
            advs, rets = self.compute_gae(rews, old_vals, next_vals, dones)
            
            advs_flat = advs.view(seq_len * self.num_agents, 1)
            rets_flat = rets.view(seq_len * self.num_agents, 1)
            
            advs_flat = (advs_flat - advs_flat.mean()) / (advs_flat.std() + 1e-8)
            
            # Old log probs
            old_dist = self.actor(obs_flat)
            if avail_flat is not None:
                old_dist.logits[~avail_flat.bool()] = -1e10
            old_log_probs = old_dist.log_prob(acts_flat.squeeze(-1)).unsqueeze(-1)
            
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
            
        for _ in range(self.ppo_epochs):
            curr_dist = self.actor(obs_flat)
            if avail_flat is not None:
                curr_dist.logits[~avail_flat.bool()] = -1e10
                
            curr_log_probs = curr_dist.log_prob(acts_flat.squeeze(-1)).unsqueeze(-1)
            entropy = curr_dist.entropy().mean()
            
            ratios = torch.exp(curr_log_probs - old_log_probs)
            
            surr1 = ratios * advs_flat
            surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advs_flat
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            
            curr_vals = self.critic(obs_flat)
            critic_loss = F.mse_loss(curr_vals, rets_flat)
            
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            if "max_grad_norm" in self.cfg["optim"]:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg["optim"]["max_grad_norm"])
            self.optimizer_actor.step()
            
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            if "max_grad_norm" in self.cfg["optim"]:
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg["optim"]["max_grad_norm"])
            self.optimizer_critic.step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.item()
            
        self.learning_steps += 1
        
        return {
            "loss/actor_loss": total_actor_loss / self.ppo_epochs,
            "loss/critic_loss": total_critic_loss / self.ppo_epochs,
            "loss/entropy": total_entropy / self.ppo_epochs
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
