from manager.ai.manager_rl import Manager_RL
from manager.ai.manager_policy_mem import Manager_Policy_Mem
from manager.ai.manager_prompt import Manager_Prompt
from stable_baselines3 import PPO
import numpy as np
import torch as th

class Manager_RL_Mem(Manager_RL):
    def __init__(self):
        policy = Manager_Policy_Mem()
        super().__init__(
            PPO(
                policy,
                policy.observation_space,
                policy.action_space,
                verbose=1
            )
        )

    def create_state(self, query_data, candidates, selected_indices):
        query_vector = self.encode_data_to_vector(query_data) 
        
        candidate_vectors = []
        for cand in candidates:
            candidate_vector = self.encode_candidate_to_vector(cand) 
            candidate_vectors.append(candidate_vector)
        candidate_vectors = np.stack(candidate_vectors) 
        
        num_candidates, _ = candidate_vectors.shape
        
        mask = np.ones((num_candidates, 1), dtype=np.float32)
        for idx in selected_indices:
            mask[idx] = 0.0  
        
        query_matrix = np.tile(query_vector, (num_candidates, 1)) 
        
        state_matrix = np.concatenate([candidate_vectors, query_matrix, mask], axis=1)
        # The shape of state_matrix is: (num_candidates, candidate_dim + query_dim + 1)
        
        # Flatten the matrix into a 1D vector.
        state_vector = state_matrix.flatten()
        return state_vector
    
    def iterative_rerank(self, candidates, n_select):
        selected_indices = []            # To store indices of chosen candidates
        available_indices = list(range(len(candidates)))
        
        while len(selected_indices) < n_select and available_indices:
            # Create state based on the remaining candidates
            state = self.create_state(candidates, selected_indices)
            
            # Use the model to predict an action.
            # The action is an index into the current (reduced) candidate list.
            action, _ = self.manager_model.predict(state, deterministic=True)
            
            # Safety: if the predicted action exceeds available indices, wrap it.
            action = action % len(available_indices)
            chosen_idx = available_indices[action]
            
            selected_indices.append(chosen_idx)
            available_indices.remove(chosen_idx)
        
        return selected_indices
        
    def run_episode_and_update(self, rounds, query_data, candidates, n_select, manager_prompt: Manager_Prompt):
        states_1, actions_1, rewards_1, dones_1 = [], [], [], []
        states_2, actions_2, rewards_2, dones_2 = [], [], [], []

        selected_indices_1, selected_indices_2 = [], []
        state_1 = self.create_state(query_data, candidates, selected_indices_1)
        state_2 = self.create_state(query_data, candidates, selected_indices_2)

        preferred_response = ""
        done = False
        while not done:
            state_tensor_1 = th.tensor(state_1, dtype=th.float32).unsqueeze(0)
            state_tensor_2 = th.tensor(state_2, dtype=th.float32).unsqueeze(0)

            action_1, value_1, log_prob_1 = self.manager_model.policy.forward(state_tensor_1, deterministic=True)
            action_2, value_2, log_prob_2 = self.manager_model.policy.forward(state_tensor_2, deterministic=True)

            action_1 = int(action_1.item())
            action_2 = int(action_2.item())

            states_1.append(state_1)
            states_2.append(state_2)
            actions_1.append(action_1)
            actions_2.append(action_2)

            selected_indices_1.append(action_1)
            selected_indices_2.append(action_2)

            if len(selected_indices_1) >= n_select and len(selected_indices_2) >= n_select:
                done = True

                final_ordering_1 = selected_indices_1.copy()
                final_ordering_2 = selected_indices_2.copy()

                response_1 = manager_prompt.generate_response(rounds, query_data, final_ordering_1)
                response_2 = manager_prompt.generate_response(rounds, query_data, final_ordering_2)

                preferred_response, reward_1, reward_2 = manager_prompt.choose_response(response_1, response_2)

            else:
                reward_1, reward_2 = 0.0, 0.0

            rewards_1.append(reward_1)
            rewards_2.append(reward_2)
            dones_1.append(done)
            dones_2.append(done)

            state_1 = self.create_state(query_data, candidates, selected_indices_1)
            state_2 = self.create_state(query_data, candidates, selected_indices_2)

        for i in range(len(states_1)):
            s1 = th.tensor(states_1[i], dtype=th.float32).unsqueeze(0)
            s2 = th.tensor(states_2[i], dtype=th.float32).unsqueeze(0)
            with th.no_grad():
                _, v1, lp1 = self.manager_model.policy.forward(s1, deterministic=True)
                _, v2, lp2 = self.manager_model.policy.forward(s2, deterministic=True)
            self.manager_model.rollout_buffer.add(s1.numpy(), actions_1[i], rewards_1[i], dones_1[i], v1.item(), lp1.item())
            self.manager_model.rollout_buffer.add(s2.numpy(), actions_2[i], rewards_2[i], dones_2[i], v2.item(), lp2.item())

        self.manager_model.train()
        return preferred_response
        