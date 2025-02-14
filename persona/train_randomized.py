import os
import signal
import threading
import random
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
    ui_root.title("Epoch Aggregated UI")
    ui_root.geometry("800x1000")
    EpochRecordKeeperUI(ui_root)
    ui_root.after(0, lambda: create_record_keeper_window(1))
    ui_root.mainloop()

def create_record_keeper_window(epoch_number):
    global current_record_keeper_window, ui_root
    if current_record_keeper_window is not None:
        try:
            current_record_keeper_window.destroy()
        except Exception as e:
            print("Error closing previous RecordKeeperUI:", e)
    current_record_keeper_window = tk.Toplevel(ui_root)
    current_record_keeper_window.title(f"Record Keeper - Epoch {epoch_number}")
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

# User selects ONE persona
while True:
    try:
        choice = int(input("Enter the number of your chosen persona: ").strip()) - 1
        if choice < 0 or choice >= len(persona_files):
            raise ValueError("Selection out of range.")
        break
    except ValueError as e:
        print("Invalid input:", e)

chosen_persona = persona_files[choice]
remaining_personas = [p for p in persona_files if p != chosen_persona]  # Remove chosen persona

print(f"\nYou have selected: {chosen_persona}")
print("Each game will randomly introduce another persona from the remaining pool.\n")

# User chooses turn count
while True:
    try:
        num_turns = int(input("Enter the number of turns per game: ").strip())
        if num_turns < 1:
            raise ValueError("The number of turns must be at least 1.")
        break
    except ValueError as e:
        print("Invalid input:", e)

# User chooses epoch count
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

    last_random_persona = None  # Track the last randomly chosen persona to avoid immediate repeats
    
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}...")

        # Select a new random persona for this epoch, ensuring it's different from the last one
        random_persona = last_random_persona
        while random_persona == last_random_persona:
            random_persona = random.choice(remaining_personas)
        last_random_persona = random_persona  # Update last used persona

        print(f"\nEpoch {epoch+1}: Your persona ({chosen_persona}) is interacting with {random_persona}.\n")

        # Create a new game instance for each epoch.
        game = Game()

        # Add the user's chosen persona as the consistent player
        chosen_npc = NPC(persona_path=os.path.join(personas_folder, chosen_persona))
        game.add_player(chosen_npc)

        # Add the randomly selected persona for this game
        npc = NPC(persona_path=os.path.join(personas_folder, random_persona))
        game.add_player(npc)

        # Play the game for the specified number of turns.
        game.play_game(num_turns)

        # Save the epoch's records and clear them for the next game.
        RecordKeeper.instance().save_epoch()
        print(f"\nEpoch {epoch+1} complete.\n")

        # Schedule UI update
        if ui_root is not None:
            ui_root.after(0, lambda epoch_num=epoch+1: create_record_keeper_window(epoch_num))

    print("Training complete. All epochs finished.")

# ---------- Start the Training Loop in its own thread ----------
training_thread = threading.Thread(target=training_loop, daemon=True)
training_thread.start()

# Run the UI in the main thread.
run_ui()
