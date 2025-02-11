class Recorder:
    def __init__(self, persona_name):
        self.persona_name = persona_name
        self.records = []

    def record(self, turn):
        self.records.append(turn)