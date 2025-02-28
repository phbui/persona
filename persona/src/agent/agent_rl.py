from agent.agent import Agent
from log.logger import Logger, Log
from manager.ai.manager_rl_mem import Manager_RL_Mem

class Agent_RL(Agent):    
    def __init__(self):
        self.name = None
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
        return ""