from log.logger import Logger, Log

class Agent:
    def __init__(self, name):
        self.name = name
        self.logger = Logger()
        self.current_chat = None
        log = Log("INFO", "agent", self.__class__.__name__, "__init__", f"Initialized agent '{name}'")
        self.logger.add_log_obj(log)
    
    def set_chat(self, chat):
        """Update the agent with the current chat instance."""
        self.current_chat = chat
        log = Log("INFO", "agent", self.__class__.__name__, "set_chat", f"Agent '{self.name}' set current chat")
        self.logger.add_log_obj(log)
    
    def generate_message(self, rounds, last_message):
        """Wait for the agent to input a message (synchronously)."""
        log = Log("INFO", "agent", self.__class__.__name__, "generate_message", f"Agent '{self.name}' queried for message")
        self.logger.add_log_obj(log)
        return ""
    
    def query_agent_for_message(self, rounds, last_message):
        """Generate a message. (Can be overridden in descendant classes.)"""
        log = Log("INFO", "agent", self.__class__.__name__, "query_agent_for_message", f"Agent '{self.name}' generating message")
        self.logger.add_log_obj(log)
        return self.generate_message(rounds, last_message)