import torch
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        """
        input_dim: Dimension of the state vector (e.g., concatenated embeddings and emotion scores)
        action_dim: Dimension of the action vector (e.g., number of mental state parameters to update)
        hidden_dim: Hidden layer size
        """
        super(Policy, self).__init__()
        # Actor network: outputs the mean of a Gaussian distribution over actions.
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        # Learnable log standard deviation for the action distribution.
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic network: estimates the value of the state.
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        """
        Returns the action mean, standard deviation, and value for the given state.
        """
        action_mean = self.actor(state)  # Shape: [batch, action_dim]
        value = self.critic(state)         # Shape: [batch, 1]
        # Expand the log_std to match action_mean dimensions and exponentiate to get std
        std = self.log_std.exp().expand_as(action_mean)
        return action_mean, std, value