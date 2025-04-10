import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class Manager_Policy(nn.Module):
    def __init__(self, input_dim, action_dim=20, num_categories=4, lr=1e-3, training=True):
        super(Manager_Policy, self).__init__()
        self.action_dim = action_dim
        self.num_categories = num_categories
        self.training = training

        # Policy Network (Actor)
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim * num_categories)
        )

        # Value Network (Critic)
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        logits = self.policy_net(state)
        if self.training:
            logits += th.randn_like(logits) * 5.0
        logits = logits.view(-1, self.action_dim, self.num_categories)
        value = self.value_net(state).squeeze(-1)
        return logits, value

    def select_action(self, state, temperature=1.0):
        logits, value = self.forward(state)
        logits = logits / temperature
        probs = Categorical(logits=logits)
        action = probs.sample()
        log_prob = probs.log_prob(action).sum(dim=-1)
        return action.squeeze(0).tolist(), log_prob, value


    def evaluate_action(self, state, action):
        """
        Compute log probability and value of a given action.
        """
        logits, value = self.forward(state)
        probs = Categorical(logits=logits)
        log_prob = probs.log_prob(action).sum(dim=-1)  # Log probs across 20 AUs
        entropy = probs.entropy().sum(dim=-1)  # Entropy across AUs
        return log_prob, entropy, value
