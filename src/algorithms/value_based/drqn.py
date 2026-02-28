import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.algorithms.base import BaseAlgorithm
from src.networks.recurrent import RecurrentQNetwork

class DRQN(BaseAlgorithm):
    """
    Deep Recurrent Q-Network.
    Processes full episodes and unrolls LSTMs to handle POMDP environments.
    """
    def __init__(self, cfg: dict, device: torch.device):
        super().__init__(cfg, device)
        
        self.obs_dim = cfg["env"]["obs_dim"]
        self.num_actions = cfg["env"]["num_actions"]
        
        self.gamma = cfg["target"]["gamma"]
        self.tau = cfg["target"]["tau"]
        self.lr = cfg["optim"]["lr"]
        self.epsilon = cfg["exploration"]["epsilon_start"]
        self.epsilon_min = cfg["exploration"]["epsilon_min"]
        self.epsilon_decay = cfg["exploration"]["epsilon_decay"]
        
        # Networks
        hidden_dims = cfg["network"]["hidden_dims"]
        rnn_hidden_dim = cfg["network"].get("rnn_hidden_dim", 64)
        
        self.q_net = RecurrentQNetwork(self.obs_dim, self.num_actions, hidden_dims, rnn_hidden_dim).to(self.device)
        self.target_net = copy.deepcopy(self.q_net).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        
        # For select_action rollout
        self.hidden_state = None
        
    def reset_hidden(self):
        """Called by env.reset() handler in train_rnn.py to reset hidden states between episodes."""
        self.hidden_state = None

    def select_action(self, obs: np.ndarray, evaluate: bool = False) -> int:
        if not evaluate and np.random.rand() < self.epsilon:
            # We still need to step the hidden state even if exploring randomly, 
            # so we do a forward pass anyway to maintain state.
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                _, self.hidden_state = self.q_net(obs_tensor, self.hidden_state)
            return np.random.randint(self.num_actions)
            
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values, self.hidden_state = self.q_net(obs_tensor, self.hidden_state)
            action = q_values.argmax(dim=-1).item()
            
        return action
        
    def _soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            
    def _hard_update(self, target, source):
        target.load_state_dict(source.state_dict())
        
    def update(self, batch: dict, steps: int) -> dict:
        # [Batch, SeqLen, Dim]
        obs = batch["obs"] 
        acts = batch["acts"].long()
        rews = batch["rews"]
        mask = batch["mask"] # 1 if valid transition, 0 if padding
        
        # In DRQN, next_obs is conceptually obs[:, 1:], and target is calculated using it.
        # But to match ReplayBuffer's output where we didn't store next_obs explicitly 
        # (since it's just shifted by 1), we compute it here:
        
        batch_size, seq_len, _ = obs.shape
        
        # We process the entire sequence to get Q values and Target Q values.
        # current_q uses obs[:, :-1], next_q uses obs[:, 1:]
        current_obs = obs[:, :-1, :]
        next_obs = obs[:, 1:, :]
        
        current_acts = acts[:, :-1, :]
        current_rews = rews[:, :-1, :]
        current_mask = mask[:, :-1, :]
        next_mask = mask[:, 1:, :] # Dones are essentially when next_mask turns 0.
        
        # Forward pass online net
        q_values, _ = self.q_net(current_obs, hidden=None)
        current_q = q_values.gather(-1, current_acts)
        
        # Forward pass target net
        with torch.no_grad():
            next_q_values, _ = self.target_net(next_obs, hidden=None)
            max_next_q = next_q_values.max(dim=-1, keepdim=True)[0]
            
            # If the next state is masked out, it means the episode ended at current_obs,
            # thus gamma * max_next_q should be 0.
            target_q = current_rews + next_mask * self.gamma * max_next_q
            
        # Calculate Masked MSE Loss
        td_error = target_q - current_q
        masked_td_error = td_error * current_mask
        
        loss = (masked_td_error ** 2).sum() / current_mask.sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        if "max_grad_norm" in self.cfg["optim"]:
            nn.utils.clip_grad_norm_(self.q_net.parameters(), self.cfg["optim"]["max_grad_norm"])
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        update_freq = self.cfg["target"].get("update_freq", 1)
        if steps % update_freq == 0:
            if self.tau < 1.0:
                self._soft_update(self.target_net, self.q_net, self.tau)
            else:
                self._hard_update(self.target_net, self.q_net)

        self.learning_steps += 1
        return {"loss/q_loss": loss.item(), "exploration/epsilon": self.epsilon}

    def save(self, filepath: str):
        torch.save(self.q_net.state_dict(), filepath)

    def load(self, filepath: str):
        self.q_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
