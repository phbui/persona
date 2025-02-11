class Player:
    def __init__(self, name):
        self.name = name

    def generate_message(self, message):
        return self.name, message