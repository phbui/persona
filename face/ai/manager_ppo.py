import os
import torch as th
import numpy as np
import torch.nn.functional as F
from ai.manager_policy import Manager_Policy


class Manager_PPO:
    def __init__(self, input_dim=777, action_dim=20, num_categories=4, gamma=0.99, clip_epsilon=0.1, gae_lambda=0.98, entropy_coef=0.5, training=True, model_path=None):
        self.model_path = model_path
        self.policy = Manager_Policy(input_dim, action_dim, num_categories, training=training) 
        self.optimizer = th.optim.Adam(self.policy.parameters(), lr=3e-4)

        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.clip_epsilon = clip_epsilon
        self.gae_lambda = gae_lambda  

        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.values, self.dones = [], [], []

    def save_model(self, save_path="models/rl/ppo_model.pth"):
        """Saves the PPO model state dictionary and optimizer."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        th.save({
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, save_path)
        #print(f"Saved PPO model to {save_path}")

    def load_model(self, load_path):
        """Loads the PPO model state dictionary and optimizer."""
        if os.path.exists(load_path):
            checkpoint = th.load(load_path, map_location="cuda" if th.cuda.is_available() else "cpu", weights_only=True)
            self.policy.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded PPO model from {load_path}")
        else:
            print("No saved PPO model found! Training a new model.")
        return self

    def store_transition(self, state, action, log_prob, reward, value, done):
        """Stores a step in the trajectory buffer"""
        self.states.append(state)
        self.actions.append(th.tensor(action)) 
        self.log_probs.append(log_prob.detach()) 
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_advantages(self):
        """Compute Generalized Advantage Estimation (GAE)"""
        returns, advantages = [], []
        last_advantage = 0

        values = self.values + [0]  # Bootstrap value at the end

        for t in reversed(range(len(self.rewards))):
            mask = 1 - self.dones[t]
            delta = self.rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            last_advantage = delta + self.gamma * self.gae_lambda * mask * last_advantage
            advantages.insert(0, last_advantage)
            returns.insert(0, last_advantage + values[t])

        return th.tensor(returns), th.tensor(advantages)

    def update_policy(self, output_dir, batch_size=32):
        """Performs a PPO update"""
        states = [th.tensor(s, dtype=th.float32) if not isinstance(s, th.Tensor) else s for s in self.states]
        states = th.stack(states)
        actions = th.stack(self.actions)  # Convert to tensor
        log_probs_old = th.tensor(self.log_probs)
        returns, advantages = self.compute_advantages()

        for _ in range(2):
            indices = th.randperm(len(states))

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
                kl = (log_probs_old[batch_indices] - new_log_probs).mean() 
                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy.mean() + 0.01 * kl
                self.policy.optimizer.zero_grad()
                loss.backward()
                total_norm = th.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                print(f"Gradient Norm: {total_norm.item()}")
                self.policy.optimizer.step()

        # Clear storage after update
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.save_model(output_dir)
