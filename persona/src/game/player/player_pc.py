from player import Player

class PC(Player):
    def __init__(self, name):
        super().__init__(name)

    def generate_message(self, history):
        return ""