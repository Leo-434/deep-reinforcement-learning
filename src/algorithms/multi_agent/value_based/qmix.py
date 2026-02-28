import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.algorithms.base import BaseAlgorithm
from src.networks.mlp import build_mlp
from src.networks.mixing import QMixer

class QMIX(BaseAlgorithm):
    """
    QMIX Algorithm.
    Centralized Training with Decentralized Execution (CTDE).
    Utilizes a mixing network to ensure monotonicity between individual Q values and Q_tot.
    """
    def __init__(self, cfg: dict, device: torch.device):
        super().__init__(cfg, device)
        
        self.num_agents = cfg["env"]["num_agents"]
        self.obs_dim = cfg["env"]["obs_dim"]
        self.state_dim = cfg["env"]["state_dim"]
        self.num_actions = cfg["env"]["num_actions"]
        
        self.gamma = cfg["target"]["gamma"]
        self.tau = cfg["target"]["tau"]
        self.lr = cfg["optim"]["lr"]
        
        self.epsilon = cfg["exploration"]["epsilon_start"]
        self.epsilon_min = cfg["exploration"]["epsilon_min"]
        self.epsilon_decay = cfg["exploration"]["epsilon_decay"]
        
        hidden_dims = cfg["network"]["hidden_dims"]
        mixing_embed_dim = cfg["network"].get("mixing_embed_dim", 32)
        
        # Individual Q-Network (Shared across all homogeneous agents)
        self.q_net = build_mlp(self.obs_dim, self.num_actions, hidden_dims).to(self.device)
        self.target_q_net = copy.deepcopy(self.q_net).to(self.device)
        
        # Mixing Network
        self.mixer = QMixer(self.num_agents, self.state_dim, mixing_embed_dim).to(self.device)
        self.target_mixer = copy.deepcopy(self.mixer).to(self.device)
        
        # Optimizer includes parameters from both Q-Network and Mixer
        self.parameters = list(self.q_net.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.lr)

    def select_action(self, obs: np.ndarray, evaluate: bool = False, avail_actions: np.ndarray = None) -> np.ndarray:
        # obs: [num_agents, obs_dim]
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
                    q_values = self.q_net(obs_tensor).squeeze(0)
                    
                    q_values[agent_avail == 0] = -9999999.0
                    action = q_values.argmax(dim=-1).item()
                    
            actions.append(action)
            
        return np.array(actions)
        
    def _soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def update(self, batch: dict, steps: int) -> dict:
        obs = batch["obs"]              # [Batch, N_Agents, ObsDim]
        states = batch["states"]        # [Batch, StateDim]
        acts = batch["acts"].long()     # [Batch, N_Agents, 1]
        rews = batch["rews"][:, 0, :]   # [Batch, 1], assumes global reward
        next_obs = batch["next_obs"]    # [Batch, N_Agents, ObsDim]
        next_states = batch["next_states"]# [Batch, StateDim]
        dones = batch["done"][:, 0, :]  # [Batch, 1], assumed global terminal
        
        avail_actions = batch.get("avail_actions", None)
        next_avail_actions = batch.get("next_avail_actions", None)
        
        batch_size = obs.shape[0]
        
        # 1. Calculate Q values for all agents
        obs_flat = obs.view(batch_size * self.num_agents, self.obs_dim)
        acts_flat = acts.view(batch_size * self.num_agents, 1)
        
        # current individual qs: [Batch*N_Agents, NumActions] -> gathered -> [Batch, N_Agents]
        q_values_flat = self.q_net(obs_flat)
        chosen_action_qvals = q_values_flat.gather(1, acts_flat).view(batch_size, self.num_agents)
        
        # 2. Mix Q values to Q_tot
        q_tot = self.mixer(chosen_action_qvals, states) # [Batch, 1]
        
        # 3. Target Q Values
        with torch.no_grad():
            next_obs_flat = next_obs.view(batch_size * self.num_agents, self.obs_dim)
            
            # True Double Q-learning: use current net to greedily select actions
            mac_out = self.q_net(next_obs_flat)
            
            if next_avail_actions is not None:
                # Need to ensure the dimensions align specifically if action spaces are heterogeneous or missing masks
                try:
                    next_avail_flat = next_avail_actions.view(batch_size * self.num_agents, self.num_actions)
                    mac_out[next_avail_flat == 0] = -9999999.0
                except RuntimeError:
                    pass # Fallback: No mask applied if dims don't match (e.g. continuous dummy masks)
                
            target_acts_flat = mac_out.max(dim=1, keepdim=True)[1]
            
            # Use target network to evaluate selected actions
            target_q_vals = self.target_q_net(next_obs_flat).gather(1, target_acts_flat).view(batch_size, self.num_agents)
            
            target_q_tot = self.target_mixer(target_q_vals, next_states) # [Batch, 1]
            targets = rews + (1 - dones) * self.gamma * target_q_tot
            
        loss = F.mse_loss(q_tot, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        if "max_grad_norm" in self.cfg["optim"]:
            nn.utils.clip_grad_norm_(self.parameters, self.cfg["optim"]["max_grad_norm"])
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        if steps % self.cfg["target"].get("update_freq", 1) == 0:
            if self.tau < 1.0:
                self._soft_update(self.target_q_net, self.q_net, self.tau)
                self._soft_update(self.target_mixer, self.mixer, self.tau)
            else:
                self.target_q_net.load_state_dict(self.q_net.state_dict())
                self.target_mixer.load_state_dict(self.mixer.state_dict())
                
        self.learning_steps += 1
        return {"loss/qmix_loss": loss.item(), "exploration/epsilon": self.epsilon}

    def save(self, filepath: str):
        torch.save({
            "q_net": self.q_net.state_dict(),
            "mixer": self.mixer.state_dict()
        }, filepath)

    def load(self, filepath: str):
        ckpt = torch.load(filepath, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.mixer.load_state_dict(ckpt["mixer"])
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
