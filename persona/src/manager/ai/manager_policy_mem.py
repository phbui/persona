import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class Policy_Mem(nn.Module):
    def __init__(self, input_dim, num_candidates, hidden_dim=64, lr=3e-4):
        """
        PPO Policy Network for iterative reranking.
        Outputs both action logits and value estimates.
        """
        super(Policy_Mem, self).__init__()
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


class Manager_Policy_Mem:
    def __init__(self, input_dim, num_candidates, gamma=0.99, clip_epsilon=0.2):
        """
        Trainer for PPO-style updates
        """
        self.policy = Policy_Mem(input_dim, num_candidates)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

        # Storage for training
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def store_transition(self, state, action, log_prob, reward, value, done):
        """Stores a step in the trajectory buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_advantages(self):
        """Compute discounted returns and advantages"""
        returns = []
        advantages = []
        last_advantage = 0
        for t in reversed(range(len(self.rewards))):
            mask = 1 - self.dones[t]
            delta = self.rewards[t] + self.gamma * self.values[t + 1] * mask - self.values[t]
            last_advantage = delta + self.gamma * self.clip_epsilon * mask * last_advantage
            advantages.insert(0, last_advantage)
            returns.insert(0, last_advantage + self.values[t])
        return th.tensor(returns), th.tensor(advantages)

    def update_policy(self, batch_size=32):
        """Performs a PPO update"""
        states = th.stack(self.states)
        actions = th.tensor(self.actions)
        log_probs_old = th.tensor(self.log_probs)
        returns, advantages = self.compute_advantages()

        for _ in range(4):  # Number of epochs
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for i in range(0, len(states), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                new_log_probs, entropy, values = self.policy.evaluate_action(batch_states, batch_actions)
                ratio = th.exp(new_log_probs - log_probs_old[batch_indices])

                surr1 = ratio * batch_advantages
                surr2 = th.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -th.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, batch_returns)

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
                self.policy.optimizer.zero_grad()
                loss.backward()
                self.policy.optimizer.step()

        # Clear storage after update
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

