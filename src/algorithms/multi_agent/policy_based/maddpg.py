import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.algorithms.base import BaseAlgorithm
from src.networks.mlp import build_mlp
from src.algorithms.policy_based.ddpg import DeterministicPolicy

class MADDPG(BaseAlgorithm):
    """
    Multi-Agent Deep Deterministic Policy Gradient (MADDPG).
    CTDE approach for continuous action spaces. 
    Actor: local obs -> local action.
    Critic: global state + joint actions -> global Q-value.
    """
    def __init__(self, cfg: dict, device: torch.device):
        super().__init__(cfg, device)
        
        self.num_agents = cfg["env"]["num_agents"]
        self.obs_dim = cfg["env"]["obs_dim"]
        self.state_dim = cfg["env"]["state_dim"]
        self.action_dim = cfg["env"]["action_dim"]
        
        if not cfg["env"].get("is_continuous", True):
            print("WARNING: MADDPG is designed for continuous action spaces.")
            
        self.max_action = cfg["env"].get("max_action", 1.0)
        
        self.gamma = cfg["target"]["gamma"]
        self.tau = cfg["target"]["tau"]
        
        self.lr_actor = cfg["optim"].get("lr_actor", 1e-4)
        self.lr_critic = cfg["optim"].get("lr_critic", 1e-3)
        self.exploration_noise = cfg["exploration"].get("noise_std", 0.1)
        
        hidden_dims = cfg["network"]["hidden_dims"]
        
        # Shared Actor (Input: Local Obs)
        self.actor = DeterministicPolicy(self.obs_dim, self.action_dim, hidden_dims, self.max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        
        # Centralized Critic
        # Input: state_dim + joint actions (num_agents * action_dim)
        critic_in_dim = self.state_dim + (self.num_agents * self.action_dim)
        
        self.critic = build_mlp(critic_in_dim, 1, hidden_dims).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
            
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)

    def select_action(self, obs: np.ndarray, evaluate: bool = False, **kwargs) -> np.ndarray:
        actions = []
        for agent_id in range(self.num_agents):
            agent_obs = obs[agent_id]
            obs_tensor = torch.as_tensor(agent_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action = self.actor(obs_tensor).cpu().numpy()[0]
                
            if not evaluate:
                noise = np.random.normal(0, self.exploration_noise * self.max_action, size=self.action_dim)
                action = np.clip(action + noise, -self.max_action, self.max_action)
                
            actions.append(action)
            
        return np.array(actions)
        
    def _soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def update(self, batch: dict, steps: int) -> dict:
        obs = batch["obs"]              # [Batch, N_Agents, ObsDim]
        states = batch["states"]        # [Batch, StateDim]
        acts = batch["acts"]            # [Batch, N_Agents, ActionDim]
        rews = batch["rews"][:, 0, :]   # [Batch, 1], assumed shared reward
        next_obs = batch["next_obs"]    # [Batch, N_Agents, ObsDim]
        next_states = batch["next_states"]# [Batch, StateDim]
        dones = batch["done"][:, 0, :]  # [Batch, 1]
        
        batch_size = obs.shape[0]
        
        # --- Flatten joint actions ---
        # Original: [Batch, N_Agents, ActionDim] -> [Batch, N_Agents * ActionDim]
        joint_acts = acts.view(batch_size, self.num_agents * self.action_dim)
        
        # --- Update Critic ---
        with torch.no_grad():
            next_obs_flat = next_obs.view(batch_size * self.num_agents, self.obs_dim)
            next_actions_flat = self.actor_target(next_obs_flat)
            # Re-form joint next actions: [Batch, N_Agents * ActionDim]
            joint_next_acts = next_actions_flat.view(batch_size, self.num_agents * self.action_dim)
            
            target_q = self.critic_target(torch.cat([next_states, joint_next_acts], dim=1))
            target_q = rews + (1 - dones) * self.gamma * target_q
            
        current_q = self.critic(torch.cat([states, joint_acts], dim=1))
        
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()
        
        # --- Update Actor ---
        obs_flat = obs.view(batch_size * self.num_agents, self.obs_dim)
        pred_actions_flat = self.actor(obs_flat)
        joint_pred_acts = pred_actions_flat.view(batch_size, self.num_agents * self.action_dim)
        
        actor_loss = -self.critic(torch.cat([states, joint_pred_acts], dim=1)).mean()
        
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
