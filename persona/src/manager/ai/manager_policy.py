import numpy as np
import torch as th
import torch.nn.functional as F
from manager.ai.policy.policy import Policy


class Manager_Policy:
    def __init__(self, input_dim, num_candidates, gamma=0.99, clip_epsilon=0.2, gae_lambda=0.95):
        """
        Trainer for PPO-style updates
        """
        self.policy = Policy(input_dim, num_candidates)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.gae_lambda = gae_lambda  # ✅ Default GAE Lambda

        # Storage for training
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def update_hyperparameters(self, lr=None, gamma=None, clip_epsilon=None, gae_param=None):
        """
        Updates PPO hyperparameters dynamically.
        """
        if lr is not None:
            self.lr = lr
            for param_group in self.policy.optimizer.param_groups:
                param_group['lr'] = self.lr  # Update optimizer learning rate

        if gamma is not None:
            self.gamma = gamma  # Update discount factor

        if clip_epsilon is not None:
            self.clip_epsilon = clip_epsilon  # Update PPO clipping range

        if gae_param is not None:
            self.gae_lambda = gae_param 

    def store_transition(self, state, action, log_prob, reward, value, done):
        """Stores a step in the trajectory buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_advantages(self):
        """Compute Generalized Advantage Estimation (GAE)"""
        returns = []
        advantages = []
        last_advantage = 0

        values = self.values + [0]  # Add bootstrap value at the end

        for t in reversed(range(len(self.rewards))):
            mask = 1 - self.dones[t]
            delta = self.rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            last_advantage = delta + self.gamma * self.gae_lambda * mask * last_advantage
            advantages.insert(0, last_advantage)
            returns.insert(0, last_advantage + values[t])

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
                batch_indices = indices[i:i + batch_size]
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
