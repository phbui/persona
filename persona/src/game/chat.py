class Chat:
    def __init__(self):
        self.history = []

    def add_turn(self, player_name, message):
        self.history.append({"order": len(self.history), "player_name": player_name, "message": message})

    def get_formatted_history(self):
        return "\n".join(f"{entry['player_name']}: {entry['message']}" for entry in self.history)