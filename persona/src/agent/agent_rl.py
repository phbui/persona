from agent.agent import Agent
from log.logger import Logger, Log
from manager.ai.manager_rl_mem import Manager_RL_Mem
from manager.ai.manager_rl_emo import Manager_RL_Emo
from manager.ai.manager_prompt import Manager_Prompt
from manager.manager_graph import Manager_Graph

class Agent_RL(Agent):    
    def __init__(self):
        self.name = None
        self.manager_graph = Manager_Graph()
        self.manager_prompt = Manager_Prompt()
        self.manager_rl_mem = Manager_RL_Mem()
        self.manager_rl_emo = Manager_RL_Emo()

    def set_name(self, name):
        self.name = name
    
    def set_hyperparameters(self, clip_range, learning_rate, discount_factor, gae_param):
        self.manager_rl_mem.set_hyperparameters(self, clip_range, learning_rate, discount_factor, gae_param)
        self.manager_rl_emo.set_hyperparameters(self, clip_range, learning_rate, discount_factor, gae_param)

    def generate_message(self, rounds, last_message):
        log = Log("INFO", "agent", self.__class__.__name__, "generate_message", f"Agent '{self.name}' queried for message")
        self.logger.add_log_obj(log)

        num_candidates = 10
        query_data, candidates = self.manager_graph.retrieve_candidates(last_message, num_candidates)

        response = self.manager_rl_mem.run_episode_and_update(rounds, query_data, candidates, num_candidates, self.name)
        return response
    
    