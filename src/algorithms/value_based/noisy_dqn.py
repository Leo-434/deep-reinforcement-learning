import copy
import torch
import torch.nn as nn
from src.algorithms.value_based.dqn import DQN
from src.networks.noisy import NoisyLinear

class NoisyMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            curr_dim = h_dim
        
        # Replace the final linear layer with NoisyLinear
        self.feature_net = nn.Sequential(*layers)
        self.noisy_out = NoisyLinear(curr_dim, output_dim)
        
    def forward(self, x: torch.Tensor):
        x = self.feature_net(x)
        return self.noisy_out(x)
        
    def reset_noise(self):
        self.noisy_out.reset_noise()


class NoisyDQN(DQN):
    """
    Noisy DQN.
    Replaces standard epsilon-greedy with NoisyLinear layers in the network.
    """
    def __init__(self, cfg: dict, device: torch.device):
        super().__init__(cfg, device)
        
        # Re-initialize networks with Noisy MLP
        hidden_dims = cfg["network"]["hidden_dims"]
        self.q_net = NoisyMLP(self.obs_dim, self.num_actions, hidden_dims).to(self.device)
        self.target_net = copy.deepcopy(self.q_net).to(self.device)
        
        # Optimizer must be updated for the new parameters
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        
        # Disable epsilon greedy entirely (noise handles it)
        self.epsilon = 0.0
        self.epsilon_min = 0.0
        
    def select_action(self, obs, evaluate=False):
        # Always use argmax, noisy layer applies randomization automatically when training
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # If evaluate=True, we should ideally disable noise, but our `.train()` vs `.eval()` 
            # mode in NoisyLinear handles this natively!
            if evaluate:
                self.q_net.eval()
            else:
                self.q_net.train()
                
            q_values = self.q_net(obs_tensor)
            action = q_values.argmax(dim=-1).item()
            
            # Reset modes depending on whether we stay evaluating
            self.q_net.train()
            
        return action
        
    def update(self, batch: dict, steps: int) -> dict:
        # Before each gradient step, we reset noise parameters in both networks
        self.q_net.reset_noise()
        self.target_net.reset_noise()
        
        # Standard DQN update logic
        return super().update(batch, steps)
