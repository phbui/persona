from persona.src.game.player.player import Player
from persona.src.game.player.persona import Persona

class NPC(Player):
    def __init__(self, persona_path):
        self.persona = Persona(persona_path)
        super().__init__(self.persona.name)

    def generate_message(self, history):
        return self.persona.generate_response(history)