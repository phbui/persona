from log.log import Log
from log.logger import Logger

class Epoch:
    def __init__(self):
        self.rounds = []
        self.logger = Logger()
        log = Log("INFO", "conversation", self.__class__.__name__, "__init__", "Created new Epoch")
        self.logger.add_log_obj(log)

    def add_round(self, round_obj):
        self.rounds.append(round_obj)
        log = Log("INFO", "conversation", self.__class__.__name__, "add_round", f"Added round: {round_obj.to_dict()}")
        self.logger.add_log_obj(log)

    def remove_round(self, index):
        if 0 <= index < len(self.rounds):
            removed = self.rounds.pop(index)
            log = Log("INFO", "conversation", self.__class__.__name__, "remove_round", f"Removed round at index {index}: {removed.to_dict()}")
            self.logger.add_log_obj(log)
            return removed
        else:
            log = Log("ERROR", "conversation", self.__class__.__name__, "remove_round", f"Invalid index: {index}")
            self.logger.add_log_obj(log)
            return None

    def get_round(self, index):
        if 0 <= index < len(self.rounds):
            return self.rounds[index]
        else:
            log = Log("ERROR", "conversation", self.__class__.__name__, "get_round", f"Invalid index: {index}")
            self.logger.add_log_obj(log)
            return None

    def get_all_rounds(self):
        return self.rounds

    def to_dict(self):
        return {"rounds": [r.to_dict() for r in self.rounds]}

    def __str__(self):
        return "\n".join([f"Round {i+1}: {str(r)}" for i, r in enumerate(self.rounds)])