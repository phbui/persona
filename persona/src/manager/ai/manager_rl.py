import numpy as np
import json

class Manager_RL():
    def __init__(self, model):
        self.manager_model = model
        self.manager_model.gamma = 0.99        # Future rewards are slightly discounted
        self.manager_model.gae_lambda = 0.95   # Balances bias and variance in advantage estimation
        self.manager_model.clip_range = 0.2    # Limits policy update step size

    def set_hyperparameters(self, clip_range, learning_rate, discount_factor, gae_param):
        for param_group in self.manager_model.policy.optimizer.param_groups:
            param_group['lr'] = learning_rate

        self.manager_model.gamma = discount_factor
        self.manager_model.gae_lambda = gae_param
        self.manager_model.clip_range = clip_range

    def encode_candidate_to_vector(self, json_obj):
        vector = self.encode_data_to_vector(json_obj)
                
        if 'semantic_score' in json_obj:
            vector.append(float(json_obj['semantic_score']))
        
        if 'bm25_score' in json_obj:
            vector.append(float(json_obj['bm25_score']))
  
        vector = np.array(vector, dtype=np.float32)
        
        desired_dim = 12
        if vector.shape[0] < desired_dim:
            vector = np.pad(vector, (0, desired_dim - vector.shape[0]), 'constant')
        else:
            vector = vector[:desired_dim]
        return vector
        
    def encode_data_to_vector(self, json_obj):
        vector = []
        
        if 'timestamp' in json_obj:
            vector.append(float(json_obj['timestamp']))
                
        if 'sentiment' in json_obj:
            try:
                sentiment_data = json.loads(json_obj['sentiment'])
                if sentiment_data and isinstance(sentiment_data, list) and len(sentiment_data) > 0:
                    label = sentiment_data[0].get('label', '').upper()
                    score = float(sentiment_data[0].get('score', 0.0))
                    polarity = 1.0 if label == "POSITIVE" else 0.0
                    vector.append(polarity)
                    vector.append(score)
                else:
                    vector.extend([0.0, 0.0])
            except Exception as e:
                vector.extend([0.0, 0.0])

        
        if 'emotion' in json_obj:
            try:
                emotion_data = json.loads(json_obj['emotion'])
                neutral_score = None
                scores = []
                for item in emotion_data:
                    score = float(item.get('score', 0.0))
                    scores.append(score)
                    if item.get('label') == 'neutral':
                        neutral_score = score
                vector.append(neutral_score if neutral_score is not None else (np.mean(scores) if scores else 0.0))
            except Exception as e:
                vector.append(0.0)
        
        vector = np.array(vector, dtype=np.float32)
        
        desired_dim = 10
        if vector.shape[0] < desired_dim:
            vector = np.pad(vector, (0, desired_dim - vector.shape[0]), 'constant')
        else:
            vector = vector[:desired_dim]
        return vector
