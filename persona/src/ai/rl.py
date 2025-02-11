from persona.src.ai.policy import Policy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

mental_change_weight = 0.6
focus_weight = 0.8
response_weight = 1.0
response_emotion_weight = 1.5

# The RL agent using PPO
class RL():
    def __init__(self, persona_name, input_dim=768 + 5, action_dim=10, hidden_dim=128, lr=3e-4, gamma=0.99, clip_epsilon=0.2):
        """
        input_dim: Dimension of the combined state vector. For example, if your sentence embedding is 768-dim
                   and you add an emotion vector of size 5, input_dim should be 768+5.
        action_dim: Dimension of the action vector (i.e., the mental state update).
        hidden_dim: Number of hidden units in the network.
        lr: Learning rate for the optimizer.
        gamma: Discount factor for future rewards.
        clip_epsilon: Clipping parameter for PPO.
        """

        self.persona_name = persona_name
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        
        # Initialize the policy network (actor and critic share the encoder)
        self.policy_net = Policy(input_dim, action_dim, hidden_dim).to("cuda")
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Buffers for storing trajectories (states, actions, log probabilities, rewards, values)
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        
    def dynamic_emotion_vector(self, emotion_results):
        # Print the raw classifier output
        print("Raw emotion classifier output:")
        print(emotion_results)
        
        # Assuming emotion_results is a list of lists, e.g.:
        # [[{'label': 'anger', 'score': 0.0044}, ...]]
        if not emotion_results or not emotion_results[0]:
            raise ValueError("No emotion results provided.")
        
        # Take the first element (i.e., results for the given input)
        emotions_list = emotion_results[0]
        print("\nEmotions list extracted (first element):")
        print(emotions_list)
        
        # Sort the list by the label to ensure a consistent order
        sorted_emotions = sorted(emotions_list, key=lambda x: x['label'])
        print("\nSorted emotions (by label):")
        print(sorted_emotions)
        
        # Extract the scores in the sorted order
        scores = [entry['score'] for entry in sorted_emotions]
        print("\nExtracted scores in sorted order:")
        print(scores)
        
        # Convert the scores list into a PyTorch tensor and move it to CUDA
        emotion_tensor = torch.tensor(scores, dtype=torch.float32).to("cuda")
        print("\nFinal emotion tensor:")
        print(emotion_tensor)
        print("Emotion tensor shape:", emotion_tensor.shape)
        
        return emotion_tensor

    # Example usage within your RL select_action method:
    def select_action(self, mental_state, embeddings, emotion_results):
        """
        Combines the previous mental state, dialogue embedding, and dynamic emotion vector into a state vector,
        performs a forward pass through the policy network,
        and samples an action from a Gaussian distribution.
        Returns the updated mental state.
        """
        # Convert dialogue embeddings to a torch tensor on CUDA
        state_embedding = torch.tensor(embeddings, dtype=torch.float32).to("cuda")
        print("\nState embedding tensor:")
        print(state_embedding)
        
        # Convert the mental state dictionary to a vector (using sorted keys)
        mental_keys = sorted(mental_state.keys())
        mental_state_vector = torch.tensor([mental_state[k] for k in mental_keys], dtype=torch.float32).to("cuda")
        print("\nMental state vector from dictionary (sorted keys):")
        print(mental_state_vector)
        
        # Dynamically extract the emotion vector using our helper function
        emotion_vector = self.dynamic_emotion_vector(emotion_results)
        
        # Concatenate the mental state vector, the dialogue embedding, and the emotion vector
        state = torch.cat([mental_state_vector, state_embedding, emotion_vector], dim=0)
        state = state.unsqueeze(0)  # Add batch dimension
        print("\nCombined state vector (with batch dimension):")
        print(state)
        
        # Forward pass through the policy network
        action_mean, std, value = self.policy_net(state)
        print("\nAction mean, std, and value from policy network:")
        print("Action mean:", action_mean)
        print("Std:", std)
        print("Value:", value)
        
        # Sample an action from the Gaussian distribution
        dist = torch.distributions.Normal(action_mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        print("\nSampled action:", action)
        print("Log probability of action:", log_prob)
        
        # Store trajectory components for policy updates
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        
        # Return the action as a NumPy array (remove batch dimension)
        return action.squeeze(0).detach().cpu().numpy()

    def update_policy(self, mental_change_reward, focus_reward, response_reward, response_emotion_reward):
        """
        Updates the policy using the collected trajectory data via a PPO update.
        The reward for the most recent step is computed by combining multiple signals.
        
        The rewards provided are scalars, which we combine (here, simply summed).
        """

        total_reward = (
            mental_change_weight * mental_change_reward +
            focus_weight * focus_reward +
            response_weight * response_reward +
            response_emotion_weight * response_emotion_reward
        )
        self.rewards.append(total_reward)
        
        # Compute discounted returns for the trajectory
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Convert stored states, actions, log probabilities, and values to tensors
        states = torch.cat(self.states, dim=0)  # Shape: [batch, input_dim]
        actions = torch.cat(self.actions, dim=0)  # Shape: [batch, action_dim]
        old_log_probs = torch.stack(self.log_probs)  # Shape: [batch]
        values = torch.cat(self.values, dim=0).squeeze()  # Shape: [batch]
        
        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        
        # Compute advantage estimates (returns - baseline values)
        advantages = returns - values.detach()
        
        # Forward pass through the policy network for all collected states
        action_means, stds, new_values = self.policy_net(states)
        dist = Normal(action_means, stds)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        
        # Compute probability ratios for PPO
        ratios = torch.exp(new_log_probs - old_log_probs)
        
        # PPO surrogate objective with clipping
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Critic loss: Mean squared error between predicted values and returns
        critic_loss = nn.MSELoss()(new_values.squeeze(), returns)
        
        # Total loss (you can add additional entropy bonuses if desired)
        loss = actor_loss + critic_loss
        
        # Perform a backpropagation step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clear trajectory buffers after update
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        
        print(f"Policy updated. Loss: {loss.item():.4f}")
