import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, input_dim, num_candidates, hidden_dim=64, lr=3e-4):
        """
        PPO Policy Network for iterative reranking.
        Outputs both action logits and value estimates.
        """
        super(Policy, self).__init__()
        self.num_candidates = num_candidates

        # Policy Network (Actor)
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_candidates)
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
        """Returns action logits and value estimate"""
        logits = self.policy_net(state)
        value = self.value_net(state).squeeze(-1)
        return logits, value

    def select_action(self, state):
        """Sample action from categorical distribution"""
        logits, value = self.forward(state)
        probs = Categorical(logits=logits)
        action = probs.sample()
        log_prob = probs.log_prob(action)
        return action.item(), log_prob, value

    def evaluate_action(self, state, action):
        """Compute log probability and value of a given action"""
        logits, value = self.forward(state)
        probs = Categorical(logits=logits)
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        return log_prob, entropy, value

