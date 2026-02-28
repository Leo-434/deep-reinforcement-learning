import torch
import torch.nn as nn
from typing import Tuple

class RecurrentQNetwork(nn.Module):
    """
    A network with MLP feature extractor + LSTM layer + MLP action heads.
    Supports processing full episodic batches [Batch, SeqLen, Dim].
    """
    def __init__(self, input_dim: int, action_dim: int, hidden_dims: list, rnn_hidden_dim: int = 64):
        super().__init__()
        
        # Feature Extractor
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            curr_dim = h_dim
        self.fc = nn.Sequential(*layers)
        
        # Recurrent Core
        self.lstm = nn.LSTM(curr_dim, rnn_hidden_dim, batch_first=True)
        
        # Action Head
        self.q_head = nn.Linear(rnn_hidden_dim, action_dim)
        self.rnn_hidden_dim = rnn_hidden_dim

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] = None):
        """
        x can be [Batch, SeqLen, Dim] or [Batch, Dim]
        """
        is_sequence = len(x.shape) == 3
        if not is_sequence:
            x = x.unsqueeze(1) # [Batch, 1, Dim]
            
        bs, seq_len, dim = x.shape
        x = x.reshape(bs * seq_len, dim)
        features = self.fc(x)
        features = features.view(bs, seq_len, -1)
        
        lstm_out, hidden_out = self.lstm(features, hidden)
        
        q_values = self.q_head(lstm_out)
        
        if not is_sequence:
            q_values = q_values.squeeze(1)
            
        return q_values, hidden_out
