import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from src.networks.mlp import build_mlp

class DiscretePolicy(nn.Module):
    """
    A standard Categorical policy for discrete action spaces.
    Outputs logits -> Softmax.
    """
    def __init__(self, input_dim: int, num_actions: int, hidden_dims=[64, 64]):
        super().__init__()
        self.net = build_mlp(input_dim, num_actions, hidden_dims)

    def forward(self, obs: torch.Tensor):
        logits = self.net(obs)
        dist = Categorical(logits=logits)
        return dist
        
class ContinuousPolicy(nn.Module):
    """
    A Standard Gaussian policy for continuous action spaces.
    Typically used in PPO/SAC. Outputs mean and log_std.
    """
    def __init__(self, input_dim: int, action_dim: int, hidden_dims=[64, 64]):
        super().__init__()
        self.net = build_mlp(input_dim, hidden_dims[-1], hidden_dims[:-1]) # feature extractor 
        
        # Heads
        self.mean_linear = nn.Linear(hidden_dims[-1], action_dim)
        
        # Can be stateless parameter or a network branch. Using parameter for simplicity matching stable-baselines
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor):
        features = self.net(obs)
        mean = self.mean_linear(features)
        std = torch.exp(self.log_std).expand_as(mean)
        
        dist = Normal(mean, std)
        return dist
