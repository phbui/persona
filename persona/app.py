import os
import threading
import signal
import tkinter as tk
from src.game.game import Game 
from src.game.player.player_pc import PC
from src.game.player.player_npc import NPC
from src.data.record_keeper_ui import RecordKeeperUI 

def signal_handler(sig, frame):
    print("Ctrl+C detected, exiting...")
    os._exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    # Locate the folder with persona JSON files
    personas_folder = os.path.join(os.path.dirname(__file__), "src", "game", "player", "personas")
    persona_files = [f for f in os.listdir(personas_folder) if f.endswith(".json")]
    if not persona_files:
        print("No persona files found in", personas_folder)
        exit(1)
    
    # Ask how many players (including the user)
    while True:
        num_players_input = input("Enter the number of players (including yourself): ").strip()
        try:
            num_players = int(num_players_input)
            if num_players < 2:
                raise ValueError("There must be at least 2 players.")
            if num_players > len(persona_files) + 1:
                raise ValueError(f"Not enough personas available. Maximum is {len(persona_files) + 1}.")
            break
        except ValueError as e:
            print("Invalid input:", e)

    print("\nAvailable Personas:")
    for idx, filename in enumerate(persona_files):
        print(f"{idx+1}: {filename}")

    selected_personas = set()  # Use a set for faster duplicate checking

    # Let the user choose (num_players - 1) personas for the NPCs
    for i in range(num_players - 1):
        while True:
            choice = input(f"Enter the number of the persona for NPC {i+1}: ").strip()
            try:
                choice_idx = int(choice) - 1
                if choice_idx < 0 or choice_idx >= len(persona_files):
                    raise ValueError("Selection out of range.")
                if persona_files[choice_idx] in selected_personas:
                    raise ValueError("This persona has already been chosen. Choose a different one.")
                break
            except ValueError as e:
                print("Invalid input:", e)

        selected_personas.add(persona_files[choice_idx])  # Add to set to ensure uniqueness
        print(f"Selected persona for NPC {i+1}: {persona_files[choice_idx]}")

    # Ask the player for their name
    player_name = input("\nEnter your name: ")

    # Ask for the number of turns
    while True:
        turns_input = input("Enter the number of turns for the game: ").strip()
        try:
            num_turns = int(turns_input)
            if num_turns < 1:
                raise ValueError("The number of turns must be at least 1.")
            break
        except ValueError as e:
            print("Invalid input:", e)

    # Initialize the game
    game = Game()
    pc = PC(player_name)

    # Add NPCs with the selected personas
    for persona_file in selected_personas:
        npc = NPC(persona_path=os.path.join(personas_folder, persona_file))
        game.add_player(npc)

    # Add the player character (PC)
    game.add_player(pc)

    # Start the game thread with the selected number of turns
    game_thread = threading.Thread(target=game.play_game, args=(num_turns,), daemon=True)
    game_thread.start()

    # Create the Record Keeper UI in a new Toplevel window
    record_keeper_window = tk.Toplevel(pc.root)
    record_keeper_window.title("Record Keeper")
    record_keeper_ui = RecordKeeperUI(record_keeper_window)

    # Start the Tkinter mainloop
    pc.root.mainloop()
