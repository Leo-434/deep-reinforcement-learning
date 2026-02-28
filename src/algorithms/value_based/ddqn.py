import torch
import torch.nn.functional as F
from src.algorithms.value_based.dqn import DQN

class DDQN(DQN):
    """
    Double Deep Q-Network (DDQN)
    Solves the overestimation bias of classic DQN by using the online network 
    to select the action and the target network to evaluate it.
    """
    def __init__(self, cfg: dict, device: torch.device):
        super().__init__(cfg, device)
        
    def update(self, batch: dict, steps: int) -> dict:
        obs = batch["obs"]
        acts = batch["acts"].long()
        rews = batch["rews"]
        next_obs = batch["next_obs"]
        dones = batch["done"]
        
        # Current Q value
        q_values = self.q_net(obs)
        current_q = q_values.gather(1, acts)
        
        # Double Q-Learning Target
        with torch.no_grad():
            # 1. Use online network to SELECT the next action
            next_actions = self.q_net(next_obs).argmax(dim=1, keepdim=True)
            
            # 2. Use target network to EVALUATE the selected action
            next_q_values = self.target_net(next_obs)
            max_next_q = next_q_values.gather(1, next_actions)
            
            target_q = rews + (1 - dones) * self.gamma * max_next_q
            
        # Loss & Optimize
        loss = F.mse_loss(current_q, target_q)
        
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
        return {"loss/q_loss": loss.item(), "exploration/epsilon": self.epsilon}
