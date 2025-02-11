import os
import threading
import signal
from src.game.game import Game 
from src.game.player.player_pc import PC
from src.game.player.player_npc import NPC

def signal_handler(sig, frame):
    print("Ctrl+C detected, exiting...")
    os._exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    personas_folder = os.path.join(os.path.dirname(__file__), "src", "game", "player", "personas")
    persona_files = [f for f in os.listdir(personas_folder) if f.endswith(".json")]
    
    if not persona_files:
        print("No persona files found in", personas_folder)
        exit(1)
    
    print("Available Personas:")
    for idx, filename in enumerate(persona_files):
        print(f"{idx+1}: {filename}")
    
    choice = input("Enter the number of the persona to use for the NPC: ").strip()
    try:
        choice_idx = int(choice) - 1
        if choice_idx < 0 or choice_idx >= len(persona_files):
            raise ValueError("Selection out of range.")
    except ValueError as e:
        print("Invalid input:", e)
        exit(1)
    
    selected_persona_path = os.path.join(personas_folder, persona_files[choice_idx])
    print("Selected persona file:", selected_persona_path)
    
    game = Game()
    npc = NPC(persona_path=selected_persona_path)
    pc = PC("Player")
    
    game.add_player(npc)
    game.add_player(pc)
    
    # Run game loop in a separate thread.
    game_thread = threading.Thread(target=game.play_game, args=(10,), daemon=True)
    game_thread.start()
    
    # Start the Tkinter event loop for the persistent chat interface.
    pc.root.mainloop()
