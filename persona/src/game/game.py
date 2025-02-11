from persona.src.game.chat import Chat

class Game:
    def __init__(self):

        self.chat = Chat()
        self.players = []
        self.turn = 0

    def add_player(self, player):
        self.add_player({"turn_order": len(self.players), "player": player})

    def play_turn(self):
        player = self.players[self.turn]

        self.chat.add_turn(player.name, 
                           player.generate_message(
                               self.chat.history))

        self.turn += 1

        if self.turn > len(self.players):
            self.turn = 0

    def play_game(self, num_turns):
        if len(self.players) > 2:
            print('Too many players!')
            return

        if len(self.players) < 2:
            print('Not enough players!')
            return
        
        turn_counter = 0

        # Player 2 is the PC
        self.chat.add_turn("[GAME WORLD]", f"{self.players[1]} approaches {self.players[0]}")

        while turn_counter < num_turns:
            self.play_turn()
            turn_counter += 1
        