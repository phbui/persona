import torch
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        """
        input_dim: Dimension of the state vector (e.g., concatenated embeddings and emotion scores)
        action_dim: Dimension of the action vector (e.g., number of mental state parameters to update)
        hidden_dim: Hidden layer size
        """
        super(Policy, self).__init__()
        # Actor network: outputs the mean of a Gaussian distribution over actions.
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        # Learnable log standard deviation for the action distribution.
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic network: estimates the value of the state.
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        """
        Returns the action mean, standard deviation, and value for the given state.
        """
        action_mean = self.actor(state)  # Shape: [batch, action_dim]
        value = self.critic(state)         # Shape: [batch, 1]
        # Expand the log_std to match action_mean dimensions and exponentiate to get std
        std = self.log_std.exp().expand_as(action_mean)
        return action_mean, std, value
    
    def __str__(self):
        # Build a detailed string describing the Policy network
        description = "Policy Network:\n"
        description += "  Actor Network (Gaussian policy):\n"
        # Access the layers in the actor network (structure: Linear -> ReLU -> Linear)
        actor_layer1 = self.actor[0]
        actor_activation = self.actor[1]
        actor_layer2 = self.actor[2]
        description += f"    - First Linear Layer: in_features={actor_layer1.in_features}, out_features={actor_layer1.out_features}\n"
        description += f"    - Activation: {actor_activation.__class__.__name__}\n"
        description += f"    - Second Linear Layer: in_features={actor_layer2.in_features}, out_features={actor_layer2.out_features}\n"
        description += f"    - Learnable log_std parameter (initial values): {self.log_std.data.tolist()}\n\n"
        
        description += "  Critic Network (State-value estimator):\n"
        # Access the layers in the critic network (structure: Linear -> ReLU -> Linear)
        critic_layer1 = self.critic[0]
        critic_activation = self.critic[1]
        critic_layer2 = self.critic[2]
        description += f"    - First Linear Layer: in_features={critic_layer1.in_features}, out_features={critic_layer1.out_features}\n"
        description += f"    - Activation: {critic_activation.__class__.__name__}\n"
        description += f"    - Second Linear Layer: in_features={critic_layer2.in_features}, out_features={critic_layer2.out_features}\n"
        
        return description
    
    def to_dict(self):
        """
        Converts the Policy network into a serializable dictionary format.
        """
        return {
            "actor": {
                "layer1": {
                    "in_features": self.actor[0].in_features,
                    "out_features": self.actor[0].out_features
                },
                "activation": self.actor[1].__class__.__name__,
                "layer2": {
                    "in_features": self.actor[2].in_features,
                    "out_features": self.actor[2].out_features
                },
                "log_std": self.log_std.data.tolist()  # Convert tensor to list
            },
            "critic": {
                "layer1": {
                    "in_features": self.critic[0].in_features,
                    "out_features": self.critic[0].out_features
                },
                "activation": self.critic[1].__class__.__name__,
                "layer2": {
                    "in_features": self.critic[2].in_features,
                    "out_features": self.critic[2].out_features
                }
            },
            "state_dict": {k: v.tolist() for k, v in self.state_dict().items()}  # Save model parameters
        }