import torch.nn as nn
from typing import List, Callable, Type

def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int],
    activation: Type[nn.Module] = nn.ReLU,
    output_activation: Type[nn.Module] = nn.Identity
) -> nn.Sequential:
    """
    Builds a Multi-Layer Perceptron (MLP).
    Args:
        input_dim: The input dimension.
        output_dim: The output dimension.
        hidden_dims: List of dimensions for hidden layers.
        activation: Activation function class (default: nn.ReLU).
        output_activation: Output layer activation function class (default: nn.Identity).
    """
    layers = []
    current_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(activation())
        current_dim = hidden_dim

    layers.append(nn.Linear(current_dim, output_dim))
    layers.append(output_activation())

    return nn.Sequential(*layers)
