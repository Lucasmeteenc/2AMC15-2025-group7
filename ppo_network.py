"""
Neural network components for PPO implementation.
Contains Actor, Critic networks and NetworkFactory.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class NetworkFactory:
    """Factory class for creating neural networks."""

    @staticmethod
    def create_actor(input_size: int, hidden_dims: int, action_dim: int, 
                    dropout: float, num_layers: int) -> "Actor":
        """Create an Actor network."""
        return Actor(input_size, hidden_dims, action_dim, dropout, num_layers)

    @staticmethod
    def create_critic(input_size: int, hidden_dims: int, dropout: float, 
                     num_layers: int) -> "Critic":
        """Create a Critic network."""
        return Critic(input_size, hidden_dims, dropout, num_layers)


class Actor(nn.Module):
    """Actor network for policy approximation in PPO."""

    def __init__(self, input_features: int, hidden_dimensions: int, action_dim: int,
                 dropout: float, num_layers: int = 3):
        super().__init__()

        if num_layers < 1:
            raise ValueError("Number of layers must be at least 1")

        self.input_features = input_features
        self.hidden_dimensions = hidden_dimensions
        self.action_dim = action_dim

        # Build dynamic network
        layers = []
        input_dim = input_features

        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dimensions),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            input_dim = hidden_dimensions

        self.backbone = nn.Sequential(*layers)
        self.action_head = nn.Linear(hidden_dimensions, action_dim)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        nn.init.orthogonal_(self.action_head.weight, gain=0.1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the actor network."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        x = self.backbone(state)
        action_logits = self.action_head(x)
        return action_logits


class Critic(nn.Module):
    """Critic network for value function approximation in PPO."""

    def __init__(self, input_features: int, hidden_dimensions: int, 
                 dropout: float, num_layers: int = 3):
        super().__init__()

        if num_layers < 1:
            raise ValueError("Number of layers must be at least 1")

        self.input_features = input_features
        self.hidden_dimensions = hidden_dimensions

        # Build dynamic network
        layers = []
        input_dim = input_features

        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dimensions),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            input_dim = hidden_dimensions

        self.backbone = nn.Sequential(*layers)
        self.value_head = nn.Linear(hidden_dimensions, 1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the critic network."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        x = self.backbone(state)
        value = self.value_head(x)
        return value