import torch
import torch.nn.functional as F
import numpy as np
from src.algorithms.value_based.dqn import DQN

class PERDQN(DQN):
    """
    Prioritized Experience Replay DQN.
    Instead of standard MSE loss, multiplies MSE by Importance Sampling weights,
    and returns absolute TD error to update priorities.
    """
    def __init__(self, cfg: dict, device: torch.device):
        super().__init__(cfg, device)
        # Note: Train loop must initialize PrioritizedReplayBuffer instead of ReplayBuffer when name="PERDQN"

    def update(self, batch: dict, steps: int) -> dict:
        obs = batch["obs"]
        acts = batch["acts"].long()
        rews = batch["rews"]
        next_obs = batch["next_obs"]
        dones = batch["done"]
        weights = batch["weights"]
        tree_idxs = batch["tree_idxs"]
        
        q_values = self.q_net(obs)
        current_q = q_values.gather(1, acts)
        
        with torch.no_grad():
            next_q_values = self.target_net(next_obs)
            max_next_q = next_q_values.max(dim=1, keepdim=True)[0]
            target_q = rews + (1 - dones) * self.gamma * max_next_q
            
        # Calculate element-wise TD Error
        td_error = target_q - current_q
        
        # Weighted MSE Loss
        loss = (weights * (td_error ** 2)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        if "max_grad_norm" in self.cfg["optim"]:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.cfg["optim"]["max_grad_norm"])
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
        
        abs_errors = torch.abs(td_error).detach().cpu().numpy().flatten()
        return {
            "loss/q_loss": loss.item(), 
            "exploration/epsilon": self.epsilon,
            "_priorities": (tree_idxs, abs_errors) # Specialized return pattern caught by train manager
        }
