import torch
import torch.nn as nn
import copy
from src.algorithms.value_based.ddqn import DDQN
from src.networks.mlp import build_mlp

class DuelingQNet(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dims: list):
        super().__init__()
        # Shared feature representation
        self.feature_net = build_mlp(input_dim, hidden_dims[-1], hidden_dims[:-1])
        
        # Value Stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, 1)
        )
        
        # Advantage Stream A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, action_dim)
        )

    def forward(self, obs: torch.Tensor):
        features = self.feature_net(obs)
        
        V = self.value_stream(features)
        A = self.advantage_stream(features)
        
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        Q = V + (A - A.mean(dim=-1, keepdim=True))
        return Q

class DuelingDQN(DDQN):
    """
    Dueling Double Deep Q-Network.
    Uses DDQN for the update rule, but overrides network initialization mathematically.
    """
    def __init__(self, cfg: dict, device: torch.device):
        # We call DDQN's init, but we overwrite self.q_net right after
        super().__init__(cfg, device)
        
        hidden_dims = cfg["network"]["hidden_dims"]
        # Overwrite parent's standard MLP with Dueling architecture
        self.q_net = DuelingQNet(self.obs_dim, self.num_actions, hidden_dims).to(device)
        self.target_net = copy.deepcopy(self.q_net).to(device)
        
        # Re-initialize optimizer for the new network parameters
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
