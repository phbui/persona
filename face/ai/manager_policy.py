import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class Manager_Policy(nn.Module):
    def __init__(self, input_dim, action_dim=20, num_categories=4, hidden_dim=128, lr=3e-4):
        """
        PPO Policy Network for Action Unit (AU) Selection.
        - `action_dim=20`: Each face has 20 AUs.
        - `num_categories=4`: Each AU can be 0,1,2,3.
        - Outputs **logits** for a 20x4 categorical distribution.
        """
        super(Manager_Policy, self).__init__()
        self.action_dim = action_dim
        self.num_categories = num_categories

        # Policy Network (Actor)
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * num_categories)  # Output (20 * 4) logits
        )

        # Value Network (Critic)
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        """Returns logits reshaped for categorical distribution & value estimate."""
        logits = self.policy_net(state)  # Shape: (batch, 80)
        logits = logits.view(-1, self.action_dim, self.num_categories)  # Reshape to (batch, 20, 4)
        value = self.value_net(state).squeeze(-1)
        return logits, value

    def select_action(self, state):
        """
        Sample an action for each AU (20 AUs, 4 categories each).
        Returns a **20D action array**, where each value is `{0,1,2,3}`.
        """
        logits, value = self.forward(state)  # (batch, 20, 4)
        probs = Categorical(logits=logits)  # Create categorical distribution
        action = probs.sample()  # Sample from the distribution (batch, 20)
        log_prob = probs.log_prob(action).sum(dim=-1)  # Sum log probs across AUs
        return action.squeeze(0).tolist(), log_prob, value  # Convert tensor to list

    def evaluate_action(self, state, action):
        """
        Compute log probability and value of a given action.
        """
        logits, value = self.forward(state)
        probs = Categorical(logits=logits)
        log_prob = probs.log_prob(action).sum(dim=-1)  # Log probs across 20 AUs
        entropy = probs.entropy().sum(dim=-1)  # Entropy across AUs
        return log_prob, entropy, value
