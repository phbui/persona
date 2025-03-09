import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class Manager_Policy(nn.Module):
    def __init__(self, input_dim, action_dim=20, num_categories=4, hidden_dim=64, lr=3e-4):
        """
        PPO Policy Network for discrete AU control.
        - Each of the 20 AUs takes values from {0,1,2,3}.
        - Outputs both action logits and value estimates.
        """
        super(Manager_Policy, self).__init__()
        self.action_dim = action_dim
        self.num_categories = num_categories  # 4 choices per AU

        # Policy Network (Actor) - Outputs logits for 80 discrete actions (20 AUs × 4 categories)
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * num_categories)  # 80 logits total
        )

        # Value Network (Critic) - Predicts state value
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        """Returns action logits and value estimate"""
        logits = self.policy_net(state)  # Shape: (batch, 80)
        logits = logits.view(-1, self.action_dim, self.num_categories)  # Reshape to (batch, 20, 4)
        value = self.value_net(state).squeeze(-1)  # Critic value
        return logits, value

    def select_action(self, state):
        """Sample a discrete 20D action vector (one value per AU)"""
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Ensure batch dimension
        logits, value = self.forward(state)

        # Sample each AU independently from categorical distribution
        probs = Categorical(logits=logits)  # Shape: (20, 4)
        action = probs.sample()  # Shape: (20,)

        log_prob = probs.log_prob(action).sum(dim=-1)  # Sum log probabilities across all 20 AUs
        return action.squeeze(0).numpy(), log_prob.detach(), value.detach()

    def evaluate_action(self, state, action):
        """Compute log probability and value of a given action"""
        logits, value = self.forward(state)
        probs = Categorical(logits=logits)
        log_prob = probs.log_prob(action).sum(dim=-1)
        entropy = probs.entropy().sum(dim=-1)
        return log_prob, entropy, value
