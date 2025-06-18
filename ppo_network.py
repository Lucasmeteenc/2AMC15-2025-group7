"""
Neural network components for PPO implementation.
Contains Actor, Critic networks and NetworkFactory.
"""

import torch
import torch.nn as nn
import numpy as np

class NetworkFactory:
    """Factory class for creating neural networks."""

    @staticmethod
    def create_actor(input_size: int, action_dim: int) -> "Actor":
        """Create an Actor network."""
        return Actor(input_size, action_dim)

    @staticmethod
    def create_critic(input_size: int) -> "Critic":
        """Create a Critic network."""
        return Critic(input_size)


class Actor(nn.Module):
    def __init__(self, input_features: int, action_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.logits_head = nn.Linear(64, action_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.)
        # output layer gets small gain (0.01) so early updates are gentle
        nn.init.orthogonal_(self.logits_head.weight, gain=0.01)

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        x = self.backbone(state)
        return self.logits_head(x)               # logits

class Critic(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.value_head = nn.Linear(64, 1)

        # Orthogonal init as in OpenAI Baselines
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.value_head(self.net(x)).squeeze(-1)
