from .policy import Policy
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

mental_change_weight = 0.6
focus_weight = 0.8
response_weight = 1.0
response_emotion_weight = 1.5

class RL():
    def __init__(self, persona_name, input_dim=768 + 6 + 7, action_dim=6, hidden_dim=128, lr=3e-4, gamma=0.99, clip_epsilon=0.2):
        """
        persona_name: Name of the persona (used to identify the policy file).
        input_dim: Dimension of the full state vector.
        action_dim: Dimension of the action vector (number of mental state attributes to update).
        hidden_dim: Hidden layer size.
        lr: Learning rate.
        gamma: Discount factor.
        clip_epsilon: PPO clipping parameter.
        """
        self.persona_name = persona_name
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

        # Initialize the policy network (actor and critic) and move it to GPU.
        self.policy_net = Policy(input_dim, action_dim, hidden_dim).to("cuda")
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Buffers for storing trajectories.
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []

        # Policy file path; stored in the 'trained' directory as persona_name.json.
        formatted_name = self.persona_name.lower().replace(" ", "_")
        self.policy_file = os.path.join(os.path.dirname(__file__), "trained", f"{formatted_name}.json")
        self.load_policy()  # Load existing policy if available, otherwise create new.

    def load_policy(self):
        """
        Checks if a policy file exists for the given persona.
        If it exists, loads the state dictionary (converting from lists to tensors).
        Otherwise, saves the initial policy.
        """
        if os.path.exists(self.policy_file):
            print(f"Loading policy from {self.policy_file}")
            with open(self.policy_file, "r") as f:
                state_dict_json = json.load(f)
            # Convert JSON lists back into tensors.
            state_dict = {}
            for key, value in state_dict_json.items():
                state_dict[key] = torch.tensor(value)
            self.policy_net.load_state_dict(state_dict)
        else:
            print(f"No policy file found for {self.persona_name}. Creating new policy.")
            self.save_policy()

    def save_policy(self):
        """
        Saves the current policy network’s state dictionary to a JSON file.
        Tensors are converted to lists for JSON serialization.
        """
        state_dict = self.policy_net.state_dict()
        state_dict_json = {}
        for key, tensor in state_dict.items():
            # Move tensor to CPU and convert to list.
            state_dict_json[key] = tensor.cpu().tolist()
        # Ensure that the directory exists.
        os.makedirs(os.path.dirname(self.policy_file), exist_ok=True)
        with open(self.policy_file, "w") as f:
            json.dump(state_dict_json, f)
        print(f"Policy saved to {self.policy_file}")

    def dynamic_emotion_vector(self, emotion_results):

        # Check if emotion_results is a dict; if so, use its items.
        if isinstance(emotion_results, dict):
            # Sort the dictionary items by key (emotion label) to ensure a consistent order.
            sorted_emotions = sorted(emotion_results.items(), key=lambda x: x[0])

            # Extract scores in sorted order.
            scores = [score for label, score in sorted_emotions]
        elif isinstance(emotion_results, list) and emotion_results:
            # If emotion_results is a list, assume its first element contains the data.
            emotions_list = emotion_results[0]
            
            # If it's a list of dicts, sort them by label.
            sorted_emotions = sorted(emotions_list, key=lambda x: x['label'])
            scores = [entry['score'] for entry in sorted_emotions]
        else:
            raise ValueError("No valid emotion results provided.")
        
        # Convert the scores list into a PyTorch tensor and move it to CUDA.
        emotion_tensor = torch.tensor(scores, dtype=torch.float32).to("cuda")
        
        return emotion_tensor

    def select_action(self, mental_state, embeddings, emotion_results):
        """
        Combines the previous mental state, dialogue embedding, and dynamic emotion vector into a state vector,
        performs a forward pass through the policy network, and samples an action (delta) for updating the mental state.
        Returns the updated mental state as a dictionary.
        """
        # Convert dialogue embeddings to a torch tensor on CUDA
        state_embedding = torch.tensor(embeddings, dtype=torch.float32).to("cuda")

        # Convert the mental state dictionary to a vector (using sorted keys)
        mental_keys = sorted(mental_state.keys())
        mental_state_vector = torch.tensor([mental_state[k] for k in mental_keys], dtype=torch.float32).to("cuda")
        
        # Dynamically extract the emotion vector using our helper function
        emotion_vector = self.dynamic_emotion_vector(emotion_results)
        
        # Concatenate the mental state vector, the dialogue embedding, and the emotion vector
        state = torch.cat([mental_state_vector, state_embedding, emotion_vector], dim=0)
        state = state.unsqueeze(0)  # Add batch dimension
        # Forward pass through the policy network
        action_mean, std, value = self.policy_net(state)
        # Sample an action from the Gaussian distribution
        dist = Normal(action_mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Store trajectory components for policy updates
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        
        # Interpret action as delta for mental state and update accordingly.
        action_delta = action.squeeze(0)  # Remove batch dimension.
        updated_mental_state_vector = mental_state_vector + action_delta
        
        updated_mental_state = {}
        for i, key in enumerate(mental_keys):
            updated_value = updated_mental_state_vector[i].item()
            updated_mental_state[key] = max(0, min(updated_value, 100))
        
        return updated_mental_state

    def update_policy(self, mental_change_reward, focus_reward, response_reward, response_emotion_reward):
        # 1. Compute the weighted total reward (using a baseline offset of 75).
        total_reward = (
            mental_change_weight * (mental_change_reward - 75) +
            focus_weight * (focus_reward - 75) +
            response_weight * (response_reward - 75) +
            response_emotion_weight * (response_emotion_reward - 75)
        )
        # print(f"Total reward: {total_reward}")
        self.rewards.append(total_reward)

        # 2. Compute discounted returns for the trajectory.
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        # print(f"Discounted returns (list): {returns}")
        returns = torch.tensor(returns, dtype=torch.float32)
        if self.values:
            returns = returns.to(self.values[0].device)
        # print(f"Discounted returns (tensor): {returns}")

        # 3. Convert stored trajectories into tensors.
        states = torch.cat(self.states, dim=0)      # Shape: [batch, input_dim]
        actions = torch.cat(self.actions, dim=0)      # Shape: [batch, action_dim]
        old_log_probs = torch.stack(self.log_probs)   # Shape: [batch]
        values = torch.cat(self.values, dim=0).squeeze()  # Shape: [batch]
        # print(f"States shape: {states.shape}")
        # print(f"Actions shape: {actions.shape}")
        # print(f"Old log_probs shape: {old_log_probs.shape}")
        # print(f"Values shape: {values.shape}")

        # 4. Normalize returns (using unbiased=False to handle single-element tensors).
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-5)
        # print(f"Normalized returns: {returns}")

        # 5. Compute advantages.
        advantages = returns - values.detach()
        # print(f"Advantages: {advantages}")

        # 6. Forward pass through the policy network.
        action_means, stds, new_values = self.policy_net(states)
        # print(f"Action means: {action_means}")
        # print(f"Standard deviations: {stds}")
        dist = Normal(action_means, stds)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        # print(f"New log_probs: {new_log_probs}")

        # 7. Compute the PPO surrogate objective.
        ratios = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        # print(f"Actor loss: {actor_loss.item()}")

        # 8. Compute the critic loss.
        critic_loss = nn.MSELoss()(new_values.squeeze(), returns)
        # print(f"Critic loss: {critic_loss.item()}")

        # 9. Total loss.
        loss = actor_loss + critic_loss
        # print(f"Total loss: {loss.item()}")

        # 10. Backpropagation step.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 11. Clear trajectory buffers after update.
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        # print("Trajectory buffers cleared.")
        # print(f"Policy updated. Loss: {loss.item():.4f}")
        self.save_policy()
