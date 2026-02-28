import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.algorithms.base import BaseAlgorithm
from src.networks.mlp import build_mlp

class IQL(BaseAlgorithm):
    """
    Independent Q-Learning (IQL).
    Each agent learns its own Q-function treating others as part of the environment.
    Implementation here shares parameters among homogeneous agents but inputs local observations.
    """
    def __init__(self, cfg: dict, device: torch.device):
        super().__init__(cfg, device)
        
        self.obs_dim = cfg["env"]["obs_dim"]
        self.num_actions = cfg["env"]["num_actions"]
        self.num_agents = cfg["env"]["num_agents"]
        
        self.gamma = cfg["target"]["gamma"]
        self.tau = cfg["target"]["tau"]
        self.lr = cfg["optim"]["lr"]
        self.epsilon = cfg["exploration"]["epsilon_start"]
        self.epsilon_min = cfg["exploration"]["epsilon_min"]
        self.epsilon_decay = cfg["exploration"]["epsilon_decay"]
        
        hidden_dims = cfg["network"]["hidden_dims"]
        
        # Shared Q-Network for all agents 
        # Output is [num_actions] taking local obs [obs_dim]
        self.q_net = build_mlp(self.obs_dim, self.num_actions, hidden_dims).to(self.device)
        self.target_net = copy.deepcopy(self.q_net).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)

    def select_action(self, obs: np.ndarray, evaluate: bool = False, avail_actions: np.ndarray = None) -> np.ndarray:
        # obs: [num_agents, obs_dim]
        # avail_actions: [num_agents, num_actions] (1 means available, 0 means unavailable)
        actions = []
        for agent_id in range(self.num_agents):
            agent_obs = obs[agent_id]
            agent_avail = avail_actions[agent_id] if avail_actions is not None else np.ones(self.num_actions)
            
            if not evaluate and np.random.rand() < self.epsilon:
                valid_acts = np.where(agent_avail == 1)[0]
                action = np.random.choice(valid_acts) if len(valid_acts) > 0 else 0
            else:
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(agent_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    q_values = self.q_net(obs_tensor).squeeze(0) # [num_actions]
                    
                    # Mask out unavailable actions
                    q_values[agent_avail == 0] = -9999999.0
                    action = q_values.argmax(dim=-1).item()
                    
            actions.append(action)
            
        return np.array(actions)
        
    def _soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def update(self, batch: dict, steps: int) -> dict:
        obs = batch["obs"]              # [Batch, N_Agents, ObsDim]
        acts = batch["acts"].long()     # [Batch, N_Agents, ActionDim]
        rews = batch["rews"]            # [Batch, N_Agents, 1]
        next_obs = batch["next_obs"]    # [Batch, N_Agents, ObsDim]
        dones = batch["done"]           # [Batch, N_Agents, 1]
        avail_actions = batch.get("avail_actions", None)           # [Batch, N_Agents, NumActions]
        next_avail_actions = batch.get("next_avail_actions", None) # [Batch, N_Agents, NumActions]
        
        batch_size = obs.shape[0]
        
        # Merge batch and agent dimensions to process all agents simultaneously
        obs_flat = obs.view(batch_size * self.num_agents, self.obs_dim)
        acts_flat = acts.view(batch_size * self.num_agents, 1) # Assumed discrete action shape [N, 1]
        rews_flat = rews.view(batch_size * self.num_agents, 1)
        dones_flat = dones.view(batch_size * self.num_agents, 1)
        next_obs_flat = next_obs.view(batch_size * self.num_agents, self.obs_dim)
        
        # Get Q values
        q_values = self.q_net(obs_flat) # [Batch * N_Agents, NumActions]
        current_q = q_values.gather(1, acts_flat)
        
        # Target Q Values
        with torch.no_grad():
            # True Double Q-learning logic:
            mac_out = self.q_net(next_obs_flat)
            if next_avail_actions is not None and next_avail_actions.numel() == batch_size * self.num_agents * self.num_actions:
                next_avail_flat = next_avail_actions.view(batch_size * self.num_agents, self.num_actions)
                mac_out[next_avail_flat == 0] = -9999999.0
                
            target_acts_flat = mac_out.max(dim=1, keepdim=True)[1]
            next_q_values = self.target_net(next_obs_flat)
            max_next_q = next_q_values.gather(1, target_acts_flat)
            
            target_q = rews_flat + (1 - dones_flat) * self.gamma * max_next_q
            
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        if "max_grad_norm" in self.cfg["optim"]:
            nn.utils.clip_grad_norm_(self.q_net.parameters(), self.cfg["optim"]["max_grad_norm"])
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        if steps % self.cfg["target"].get("update_freq", 1) == 0:
            if self.tau < 1.0:
                self._soft_update(self.target_net, self.q_net, self.tau)
            else:
                self.target_net.load_state_dict(self.q_net.state_dict())

        self.learning_steps += 1
        return {"loss/q_loss": loss.item(), "exploration/epsilon": self.epsilon}

    def save(self, filepath: str):
        torch.save(self.q_net.state_dict(), filepath)

    def load(self, filepath: str):
        self.q_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
