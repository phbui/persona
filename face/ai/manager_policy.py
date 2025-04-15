import math
import random
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

        self.step = 0
        self.epsilon_start = 1.0
        self.epsilon_end = 0.02  
        self.epsilon_decay_steps = 500 
        self.base_temperature = 0.5 

        # Improved network architecture
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, action_dim * num_categories)
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1))

        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)

    def forward(self, state):
        logits = self.policy_net(state)
        if self.training:
            noise = th.randn_like(logits) * 0.5
            logits = logits + noise * self.get_epsilon()
        
        logits = logits.view(-1, self.action_dim, self.num_categories)
        value = self.value_net(state).squeeze(-1)
        if self.training:
            value += th.randn_like(value) * 0.2
        
        return logits, value

    def get_epsilon(self):
        progress = min(self.step / self.epsilon_decay_steps, 1.0)
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (1 - progress)**3
        return max(epsilon, self.epsilon_end)

    def select_action(self, state, temperature=None):
        dynamic_temp = temperature if temperature else max(2.0 - 0.1*self.step, 0.3)
        logits, value = self.forward(state)
        temp_scaled_logits = logits / dynamic_temp
        dist = Categorical(logits=temp_scaled_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        epsilon = self.get_epsilon()
        if self.training and random.random() < epsilon:
            noise = th.randn_like(logits) * 2.0
            noisy_logits = logits + noise * epsilon
            noisy_dist = Categorical(logits=noisy_logits/dynamic_temp)
            action = noisy_dist.sample()
            log_prob = noisy_dist.log_prob(action).sum(dim=-1)

        self.step += 1
        return action.squeeze(0).tolist(), log_prob, value

    def evaluate_action(self, state, action):
        logits, value = self.forward(state)
        probs = Categorical(logits=logits)
        log_prob = probs.log_prob(action).sum(dim=-1)
        entropy = probs.entropy().mean()  
        return log_prob, entropy, value