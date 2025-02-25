from agent import Agent
from log.logger import Logger, Log

class Agent_RL(Agent):    

    

    def generate_message(self, rounds, last_message):
        log = Log("INFO", "agent", self.__class__.__name__, "generate_message", f"Agent '{self.name}' queried for message")
        self.logger.add_log_obj(log)
        return ""