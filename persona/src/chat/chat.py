from log.logger import Logger, Log
from .round import Round 

class Chat:
    def __init__(self, agent_a, agent_b, rounds_count):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.rounds_count = rounds_count
        self.rounds = []  # List to store Round objects
        self.logger = Logger()
        log = Log("INFO", "chat", self.__class__.__name__, "__init__", 
                  f"Chat initiated with agents '{agent_a.name}' and '{agent_b.name}' for {rounds_count} rounds")
        self.logger.add_log_obj(log)
    
    def run(self):
        """Run a chat session for the specified number of rounds."""
        # Let agents know about the current chat (up-to-date conversation)
        self.agent_a.set_chat(self)
        self.agent_b.set_chat(self)

        last_message = f"{self.agent_b.name} approaches {self.agent_a.name}"
        
        for i in range(self.rounds_count):
            log = Log("INFO", "chat", self.__class__.__name__, "run", f"Starting round {i+1}")
            self.logger.add_log_obj(log)
            
            # Agent A generates and sends a message.
            message_a = self.agent_a.query_agent_for_message(self.rounds, last_message)
            message_b = self.agent_b.query_agent_for_message(self.rounds, message_a)
            last_message = message_b

            # Create and store the round.
            round_obj = Round(self.agent_a.name, message_a, self.agent_b.name, message_b)
            self.rounds.append(round_obj)
            
            log_round = Log("INFO", "chat", self.__class__.__name__, "run", 
                            f"Completed round {i+1}: {round_obj.to_dict()}")
            self.logger.add_log_obj(log_round)
            
        return self.rounds
    
    def get_rounds(self):
        """Return the rounds generated in this chat session."""
        return self.rounds
