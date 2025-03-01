import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
import numpy as np

class Manager_Policy_Mem(ActorCriticPolicy):
    num_candidates = 10      # fixed number of candidates
    candidate_dim = 12       # candidate features
    query_dim = 10           # query features
    mask_dim = 1             # mask dimension
    total_feature_dim = candidate_dim + query_dim + mask_dim  # 23
    obs_dim = num_candidates * total_feature_dim  
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    action_space = spaces.Discrete(num_candidates)  

    def __init__(self, **kwargs):
        super(Manager_Policy_Mem, self).__init__(observation_space=self.observation_space, action_space=self.action_space, **kwargs)
        # Define a simple MLP that processes each candidate independently.
        # It will output one logit per candidate.
        self.policy_net = nn.Sequential(
            nn.Linear(self.total_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # outputs a single logit for this candidate
        )
        
        # A simple value network operating on the full flattened observation.
        self.value_net = nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Set the action distribution to Categorical (Discrete actions).
        self.action_dist = th.distributions.Categorical

    def forward(self, obs, deterministic=False):
        # obs: tensor of shape (batch_size, obs_dim)
        batch_size = obs.shape[0]
        # Reshape to (batch_size, num_candidates, total_feature_dim)
        obs_reshaped = obs.view(batch_size, self.num_candidates, self.total_feature_dim)
        
        # Extract the mask from the last dimension: shape (batch_size, num_candidates)
        mask = obs_reshaped[:, :, -1]  # candidates' mask
        
        # Flatten candidate-level observations for processing:
        candidate_features = obs_reshaped.view(batch_size * self.num_candidates, self.total_feature_dim)
        
        # Get logits per candidate: shape (batch_size * num_candidates, 1)
        logits = self.policy_net(candidate_features)
        logits = logits.view(batch_size, self.num_candidates)  # shape: (batch_size, num_candidates)
        
        # Apply the mask: convert mask to log-space.
        # For mask==1, log(1+eps) ~ 0; for mask==0, log(0+eps) becomes very negative.
        eps = 1e-8
        mask_tensor = mask.float()  # shape (batch_size, num_candidates)
        masked_logits = logits + (mask_tensor + eps).log()
        
        # Create a categorical distribution from masked logits.
        distribution = self.action_dist(logits=masked_logits)
        
        # Choose actions either deterministically or via sampling.
        if deterministic:
            actions = th.argmax(masked_logits, dim=1)
        else:
            actions = distribution.sample()
            
        # Get log probabilities of selected actions.
        log_prob = distribution.log_prob(actions)
        # Compute value estimate from the entire flattened observation.
        values = self.value_net(obs)
        
        return actions, values, log_prob

    def predict(self, obs, deterministic=False):
        actions, _, _ = self.forward(obs, deterministic=deterministic)
        return actions
    
    def train(self, rollout_buffer):
        advantages, returns = self.compute_advantages(rollout_buffer)

        for _ in range(self.epochs):
            for start in range(0, len(rollout_buffer.states), self.batch_size):
                end = start + self.batch_size

                states = th.tensor(rollout_buffer.states[start:end], dtype=th.float32)
                actions = th.tensor(rollout_buffer.actions[start:end], dtype=th.int64)
                old_log_probs = th.tensor(rollout_buffer.log_probs[start:end], dtype=th.float32)
                returns_batch = th.tensor(returns[start:end], dtype=th.float32)
                advantages_batch = th.tensor(advantages[start:end], dtype=th.float32)

                _, values, log_probs = self.forward(states)
                values = values.squeeze()
                log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()

                ratio = th.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages_batch
                surr2 = th.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages_batch
                policy_loss = -th.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, returns_batch)
                entropy_loss = -self.ent_coef * self.action_dist(logits=log_probs).entropy().mean()

                loss = policy_loss + self.vf_coef * value_loss + entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def compute_advantages(self, rollout_buffer):
        advantages = np.zeros_like(rollout_buffer.rewards, dtype=np.float32)
        returns = np.zeros_like(rollout_buffer.rewards, dtype=np.float32)
        last_advantage = 0.0
        for t in reversed(range(len(rollout_buffer.rewards))):
            mask = 1.0 - rollout_buffer.dones[t]
            delta = rollout_buffer.rewards[t] + self.gamma * rollout_buffer.values[t + 1] * mask - rollout_buffer.values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * mask * last_advantage
        returns = advantages + rollout_buffer.values[:-1]
        return advantages, returns