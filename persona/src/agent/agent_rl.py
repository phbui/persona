from agent.agent import Agent
from log.logger import Logger, Log
from manager.ai.manager_rl_mem import Manager_RL_Mem
from manager.ai.manager_rl_emo import Manager_RL_Emo
from manager.ai.manager_prompt import Manager_Prompt
from manager.manager_graph import Manager_Graph

class Agent_RL(Agent):    
    candidate_dim = 12  # Features per candidate
    query_dim = 10      # Features from the query
    mask_dim = 1        # Mask indicating if a candidate was selected
    total_feature_dim = candidate_dim + query_dim + mask_dim  # 12 + 10 + 1 = 23
    num_candidates = 10  # Number of candidates considered per step
    obs_dim = num_candidates * total_feature_dim  # 10 * 23 = 230


    def __init__(self):
        self.name = None
        self.manager_graph = Manager_Graph()
        self.manager_prompt = Manager_Prompt()
        self.manager_rl_mem = Manager_RL_Mem(self.obs_dim, self.num_candidates)
        self.manager_rl_emo = Manager_RL_Emo(self.obs_dim, self.num_candidates)

    def set_name(self, name):
        self.name = name
    
    def set_hyperparameters(self, clip_range, learning_rate, discount_factor, gae_param):
        self.manager_rl_mem.set_hyperparameters(self, clip_range, learning_rate, discount_factor, gae_param)
        self.manager_rl_emo.set_hyperparameters(self, clip_range, learning_rate, discount_factor, gae_param)

    def generate_message(self, rounds, last_message):
        log = Log("INFO", "agent", self.__class__.__name__, "generate_message", f"Agent '{self.name}' queried for message")
        self.logger.add_log_obj(log)

        query_data, candidates = self.manager_graph.retrieve_candidates(last_message, self.num_candidates)

        response = self.manager_rl_mem.run_episode_and_update(rounds, query_data, candidates, self.num_candidates, self.name)
        return response
    
    