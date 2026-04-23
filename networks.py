import torch
import torch.nn as nn

# Maps state to a bounded mean action for each joint.
# LayerNorm stabilises activations across the wide state space.
# Last layer uses small weights so early actions are near-zero.
class Actor(nn.Module):
    def __init__(self, state_dim=39, action_dim=12, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        nn.init.orthogonal_(self.net[-2].weight, gain=0.01)
        nn.init.zeros_(self.net[-2].bias)

    def forward(self, state):
        return self.net(state)

# Estimates expected future return from a given state.
# Wider hidden dim than actor; LayerNorm for the same stability reasons.
class Critic(nn.Module):
    def __init__(self, state_dim=39, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.net(state)
