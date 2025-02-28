from manager.ai.manager_model import Manager_Model
from manager.ai.manager_rl import Manager_RL
import numpy as np

class Manager_RL_Emo(Manager_RL):
    def __init__(self, model):
        self.manager_model = model
    
    def create_state(self, query_data, candidates, response_data, selected_indices):
        query_vector = self.encode_data_to_vector(query_data) 
        response_vector = self.encode_data_to_vector(response_data) 
        
        candidate_vectors = []
        for cand in candidates:
            candidate_vector = self.encode_candidate_to_vector(cand) 
            candidate_vectors.append(candidate_vector)
        candidate_vectors = np.stack(candidate_vectors) 
        
        num_candidates, _ = candidate_vectors.shape
        
        mask = np.ones((num_candidates, 1), dtype=np.float32)
        for idx in selected_indices:
            mask[idx] = 0.0  
        
        query_response_vector = np.concatenate([query_vector, response_vector]) 
        query_matrix = np.tile(query_response_vector, (num_candidates, 1)) 
        
        state_matrix = np.concatenate([candidate_vectors, query_matrix, mask], axis=1)
        # The shape of state_matrix is: (num_candidates, candidate_dim + query_dim + 1)
        
        # Flatten the matrix into a 1D vector.
        state_vector = state_matrix.flatten()
        return state_vector