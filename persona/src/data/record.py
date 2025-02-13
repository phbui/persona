from .record_keeper import RecordKeeper

class Record:
    def __init__(self, persona_name):
        self.persona_name = persona_name
        self.records = []
        RecordKeeper.instance().register(self)

    def record(self, turn):
        self.records.append(turn)

    def to_dict(self):
        return {
            "persona_name": self.persona_name,
            "records": [turn.to_dict() for turn in self.records]
        }
