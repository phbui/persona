from log.log import Log
from log.logger import Logger

class Round:
    def __init__(self, speaker_a, message_a, speaker_b, message_b):
        self.speaker_a = speaker_a
        self.message_a = message_a
        self.speaker_b = speaker_b
        self.message_b = message_b
        self.logger = Logger()
        log = Log("INFO", "conversation", self.__class__.__name__, "__init__", f"Created Round: {self.to_dict()}")
        self.logger.add_log_obj(log)

    def to_dict(self):
        return {
            "speaker_a": self.speaker_a,
            "message_a": self.message_a,
            "speaker_b": self.speaker_b,
            "message_b": self.message_b
        }

    def __str__(self):
        return f"{self.speaker_a}: {self.message_a} | {self.speaker_b}: {self.message_b}"