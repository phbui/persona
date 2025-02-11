class Player:
    def __init__(self, name):
        self.name = name

    def connect_recorder(self, recorder):
        self.recorder = recorder

    def generate_message(self, history):
        message = "PLAYER CLASS MESSAGE"
        return message