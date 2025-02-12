class Chat:
    def __init__(self):
        self.history = []

    def add_turn(self, player_name, message):
        # print(f"{player_name}: {message}")
        self.history.append({"order": len(self.history), "player_name": player_name, "message": message})