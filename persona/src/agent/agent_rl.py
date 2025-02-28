from agent.agent import Agent
from log.logger import Logger, Log
from manager.ai.manager_model import Manager_Model

class Agent_RL(Agent):    
    def __init__(self, name):
        super.__init__(name)


    def generate_message(self, rounds, last_message):
        log = Log("INFO", "agent", self.__class__.__name__, "generate_message", f"Agent '{self.name}' queried for message")
        self.logger.add_log_obj(log)
        return ""