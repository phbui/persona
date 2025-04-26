from log.logger import Logger, Log
from chat.chat import Chat
from chat.epoch import Epoch  

class Manager_Chat:
    def __init__(self, agent_a, agent_b, num_epochs, rounds_per_epoch):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.num_epochs = num_epochs
        self.rounds_per_epoch = rounds_per_epoch
        self.epochs = [] 
        self.logger = Logger()
        log = Log("INFO", "manager_chat", self.__class__.__name__, "__init__", 
                  f"Manager_Chat initialized with agents '{agent_a.name}', '{agent_b.name}', "
                  f"{num_epochs} epochs, {rounds_per_epoch} rounds per epoch")
        self.logger.add_log_obj(log)
    
    def run(self):
        """Run the full conversation over the specified number of epochs."""
        for epoch_index in range(self.num_epochs):
            log_epoch_start = Log("INFO", "manager_chat", self.__class__.__name__, "run", 
                                  f"Starting epoch {epoch_index+1}")
            self.logger.add_log_obj(log_epoch_start)
            
            # Create a new chat session for the current epoch.
            chat_session = Chat(self.agent_a, self.agent_b, self.rounds_per_epoch)
            rounds = chat_session.run()
            
            # Create an Epoch object and add all rounds to it.
            epoch = Epoch()
            for round_obj in rounds:
                epoch.add_round(round_obj)
            
            self.epochs.append(epoch)
            
            log_epoch_end = Log("INFO", "manager_chat", self.__class__.__name__, "run", 
                                f"Completed epoch {epoch_index+1}")
            self.logger.add_log_obj(log_epoch_end)
        
        return self.get_conversation_history()
    
    def get_conversation_history(self):
        """Return the conversation history as a list of epoch dictionaries."""
        history = [epoch.to_dict() for epoch in self.epochs]
        log = Log("INFO", "manager_chat", self.__class__.__name__, "get_conversation_history", 
                  "Retrieved full conversation history")
        self.logger.add_log_obj(log)
        return history
