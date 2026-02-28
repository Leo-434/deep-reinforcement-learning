import torch
import torch.nn as nn
import torch.nn.functional as F

class VDNMixer(nn.Module):
    """
    Value Decomposition Networks (VDN) Mixer.
    Simply sums the individual Q-values.
    """
    def __init__(self):
        super().__init__()

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor = None):
        # agent_qs shape: [batch_size, num_agents]
        return torch.sum(agent_qs, dim=-1, keepdim=True)


class QMixer(nn.Module):
    """
    QMIX Mixer network.
    Uses the global state to generate weights and biases for mixing individual Q values.
    Weights are constrained to be non-negative (via absolute value) to ensure monotonicity.
    """
    def __init__(self, num_agents: int, state_dim: int, mixing_embed_dim: int = 32):
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.embed_dim = mixing_embed_dim

        # Hypernetwork for weights w1
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, num_agents * mixing_embed_dim)
        )

        # Hypernetwork for weights w2
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, mixing_embed_dim)
        )

        # Hypernetwork for biases b1 (note: no non-negativity constraint on bias)
        self.hyper_b1 = nn.Linear(state_dim, mixing_embed_dim)
        
        # Hypernetwork for final bias b2
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1)
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor):
        # agent_qs: [batch_size, num_agents] -> [batch_size, 1, num_agents]
        batch_size = agent_qs.size(0)
        agent_qs = agent_qs.view(batch_size, 1, self.num_agents)
        
        # w1: [batch_size, num_agents, embed_dim]
        w1 = torch.abs(self.hyper_w1(states))
        w1 = w1.view(batch_size, self.num_agents, self.embed_dim)
        
        # b1: [batch_size, 1, embed_dim]
        b1 = self.hyper_b1(states).view(batch_size, 1, self.embed_dim)
        
        # Hidden layer: [batch_size, 1, embed_dim]
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        
        # w2: [batch_size, embed_dim, 1]
        w2 = torch.abs(self.hyper_w2(states))
        w2 = w2.view(batch_size, self.embed_dim, 1)
        
        # b2: [batch_size, 1, 1]
        b2 = self.hyper_b2(states).view(batch_size, 1, 1)
        
        # Output: [batch_size, 1, 1] -> [batch_size, 1]
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(batch_size, 1)
