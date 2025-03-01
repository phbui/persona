import torch as th
import numpy as np
from manager.ai.manager_policy_mem import Manager_Policy_Mem
from manager.ai.manager_prompt import Manager_Prompt

class Manager_RL_Mem:
    def __init__(self, input_dim, num_candidates):
        """Manages the iterative reranking and training process"""
        self.ppo = Manager_Policy_Mem(input_dim, num_candidates)

    def create_state(self, query_vector, candidates, selected_indices):
        """Encodes query and candidate features into state representation"""
        candidate_vectors = np.array([self.encode_candidate_to_vector(c) for c in candidates])
        mask = np.ones((len(candidates), 1), dtype=np.float32)
        for idx in selected_indices:
            mask[idx] = 0.0

        query_matrix = np.tile(query_vector, (len(candidates), 1))
        state_matrix = np.concatenate([candidate_vectors, query_matrix, mask], axis=1)
        return state_matrix.flatten()

    def iterative_rerank(self, candidates, n_select):
        """Selects candidates through iterative reranking"""
        selected_indices = []
        available_indices = list(range(len(candidates)))

        while len(selected_indices) < n_select and available_indices:
            state_vector = self.create_state(candidates, selected_indices)
            state_tensor = th.tensor(state_vector, dtype=th.float32).unsqueeze(0)
            action, log_prob, value = self.ppo.policy.select_action(state_tensor)

            action = action % len(available_indices)
            chosen_idx = available_indices[action]

            selected_indices.append(chosen_idx)
            available_indices.remove(chosen_idx)

            self.ppo.store_transition(state_tensor, action, log_prob, 0, value, done=False)

        return selected_indices

    def run_episode_and_update(self, query_data, candidates, n_select, manager_prompt: Manager_Prompt, agent_name):
        """Runs two-stage reranking and trains PPO"""
        selected_indices_1 = self.iterative_rerank(candidates, n_select)
        selected_indices_2 = self.iterative_rerank(candidates, n_select)

        response_1 = manager_prompt.generate_response(query_data, selected_indices_1, agent_name)
        response_2 = manager_prompt.generate_response(query_data, selected_indices_2, agent_name)

        preferred_response, reward_1, reward_2 = manager_prompt.choose_response(query_data, response_1, response_2)

        # Assign rewards to each selection step
        for i in range(len(selected_indices_1)):
            self.ppo.rewards[-(i+1)] = reward_1  # Assign reward backward
        for i in range(len(selected_indices_2)):
            self.ppo.rewards[-(i+1)] = reward_2  # Assign reward backward

        # Finalize advantage calculations and update policy
        self.ppo.values.append(0)  # Bootstrap last value
        self.ppo.update_policy()

        return preferred_response
