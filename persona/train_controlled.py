import os
import signal
import threading
import tkinter as tk

from src.game.game import Game 
from src.game.player.player_npc import NPC
from src.data.record_keeper import RecordKeeper
from src.data.record_keeper_ui import RecordKeeperUI
from src.data.epoch_keeper_ui import EpochRecordKeeperUI

def signal_handler(sig, frame):
    print("Ctrl+C detected, exiting...")
    os._exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Global variables for the UI root and the currently open RecordKeeperUI window.
ui_root = None
current_record_keeper_window = None

def run_ui():
    global ui_root
    ui_root = tk.Tk()
    # Use the main root as the aggregated Epoch Keeper UI.
    ui_root.title("Epoch Aggregated UI")
    ui_root.geometry("800x1000")
    # Instantiate the aggregated UI on the main root.
    EpochRecordKeeperUI(ui_root)
    ui_root.after(0, lambda: create_record_keeper_window(1))
    ui_root.mainloop()


def create_record_keeper_window(epoch_number):
    global current_record_keeper_window, ui_root
    # Close previous RecordKeeperUI window if it exists.
    if current_record_keeper_window is not None:
        try:
            current_record_keeper_window.destroy()
        except Exception as e:
            print("Error closing previous RecordKeeperUI:", e)
    # Create a new Toplevel window for this epoch.
    current_record_keeper_window = tk.Toplevel(ui_root)
    current_record_keeper_window.title(f"Record Keeper - Epoch {epoch_number}")
    # Instantiate the RecordKeeperUI (which shows logs and analysis for that epoch).
    RecordKeeperUI(current_record_keeper_window)

# ---------- Gather Training Parameters ----------
personas_folder = os.path.join(os.path.dirname(__file__), "src", "game", "player", "personas")
persona_files = [f for f in os.listdir(personas_folder) if f.endswith(".json")]
if not persona_files:
    print("No persona files found in", personas_folder)
    exit(1)

print("Available Personas:")
for idx, filename in enumerate(persona_files):
    print(f"{idx+1}: {filename}")

while True:
    try:
        num_personas = int(input("Enter the number of personas to use for NPCs: ").strip())
        if num_personas < 2:
            raise ValueError("You must select at least 2 personas.")
        if num_personas > len(persona_files):
            raise ValueError(f"Not enough persona files available (max {len(persona_files)}).")
        break
    except ValueError as e:
        print("Invalid input:", e)

selected_personas = set()
for i in range(num_personas):
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
    selected_personas.add(persona_files[choice_idx])
    print(f"Selected persona for NPC {i+1}: {persona_files[choice_idx]}")

while True:
    try:
        num_turns = int(input("Enter the number of turns per game: ").strip())
        if num_turns < 1:
            raise ValueError("The number of turns must be at least 1.")
        break
    except ValueError as e:
        print("Invalid input:", e)

while True:
    try:
        num_epochs = int(input("Enter the number of epochs (games) for training: ").strip())
        if num_epochs < 1:
            raise ValueError("The number of epochs must be at least 1.")
        break
    except ValueError as e:
        print("Invalid input:", e)

# ---------- Define the Training Loop Function ----------
def training_loop():
    print("\n--- Starting Training ---\n")

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}...")
        # Create a new game instance for each epoch.
        game = Game()
        # Add NPCs (each with its chosen persona).
        for persona_file in selected_personas:
            npc = NPC(persona_path=os.path.join(personas_folder, persona_file))
            game.add_player(npc)
        
        # Play the game for the specified number of turns.
        game.play_game(num_turns)
        
        # Save the epoch's records and clear them for the next game.
        RecordKeeper.instance().save_epoch()
        print(f"Epoch {epoch+1} complete.\n")
        
        # Schedule creation of a new RecordKeeperUI window for this epoch on the UI thread.
        if ui_root is not None:
            ui_root.after(0, lambda epoch_num=epoch+1: create_record_keeper_window(epoch_num))
    
    print("Training complete. All epochs finished.")

# ---------- Start the Training Loop in its own thread ----------
training_thread = threading.Thread(target=training_loop, daemon=True)
training_thread.start()

# Run the UI in the main thread.
run_ui()
