import tkinter as tk
from tkinter import ttk
import os
import sys
import glob
import json
import argparse

# Import your Chat class.
# (Ensure that Chat is a Toplevel subclass for multi-window management.)
from chat import Chat
from dm_engine import Persona  # Ensure Persona is imported

class Desktop(tk.Tk):
    def __init__(self, hf_key, personas_folder):
        super().__init__()
        self.title("Dungeon Master Desktop")
        self.geometry("400x600")
        self.hf_key = hf_key
        self.personas_folder = personas_folder

        print("[DEBUG] Desktop initialized with hf_key:", hf_key)
        print("[DEBUG] Using personas folder:", self.personas_folder)

        # Bind the on_close method to the window's close protocol.
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Instead of a Listbox, use a Treeview for better control over item appearance.
        self.style = ttk.Style()
        self.style.configure("Treeview", rowheight=30)
        self.contacts_tree = ttk.Treeview(self, columns=("Name",), show="headings", selectmode="browse")
        self.contacts_tree.heading("Name", text="Contact")
        self.contacts_tree.column("Name", anchor="center")
        self.contacts_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Load persona files from the personas folder.
        self.contacts = self.load_personas()
        print("[DEBUG] Loaded", len(self.contacts), "contacts.")
        for contact in self.contacts:
            print("[DEBUG] Loaded contact:", contact['name'].strip(), "from file:", contact['file'])
            self.contacts_tree.insert("", tk.END, values=(contact['name'],))

        # Bind double-click event to open a Chat window for the selected contact.
        self.contacts_tree.bind("<Double-Button-1>", self.open_chat_with_contact)

    def load_personas(self):
        """Load all persona JSON files from the personas folder and return a list of contacts.
           Each contact includes the loaded Persona object."""
        persona_files = glob.glob(os.path.join(self.personas_folder, "*.json"))
        print("[DEBUG] Found", len(persona_files), "persona files in folder.")
        contacts = []
        for pfile in persona_files:
            print("[DEBUG] Loading persona file:", pfile)
            try:
                with open(pfile, "r", encoding="utf-8") as f:
                    data = json.load(f)
                name = data.get("username") or data.get("name") or os.path.basename(pfile).split(".")[0]
                # Create a Persona object from the file.
                persona_obj = Persona(pfile)
                contacts.append({
                    "name": name,
                    "file": pfile,
                    "persona": persona_obj
                })
                print("[DEBUG] Successfully loaded persona:", name)
            except Exception as e:
                print(f"[DEBUG] Error loading {pfile}: {e}")
        return contacts

    def open_chat_with_contact(self, event):
        selected_item = self.contacts_tree.focus()
        if selected_item:
            index = self.contacts_tree.index(selected_item)
            contact = self.contacts[index]
            persona_path = contact["file"]
            persona_obj = contact["persona"]
            print("[DEBUG] Opening Chat window for contact:", contact["name"], "from file:", persona_path)
            try:
                self.contacts_tree.state(["disabled"])
                print("[DEBUG] Contacts widget disabled.")
            except Exception as e:
                print(f"[DEBUG] Could not disable contacts widget: {e}")
            
            # Pass self (Desktop instance) as the parent to the Chat window.
            # Pass the loaded Persona object instead of a persona_path.
            chat_window = Chat(self, self.hf_key, persona_obj, max_tokens=64)
            chat_window.lift()
            self.wait_window(chat_window)
            try:
                #print("[DEBUG] Chat window closed. Conversation end data:", chat_window.conversation_end_data)
                persona_obj.append_to_backstory(chat_window.conversation_end_data)
                #print("[DEBUG] Player model end data:", chat_window.player_model_end_data)
                print("[DEBUG] Conversation saved.")
            except Exception as e:
                print("[DEBUG] No relevant info from conversation.")

            try:
                self.contacts_tree.state(["!disabled"])
                print("[DEBUG] Contacts widget re-enabled.")
            except Exception as e:
                print(f"[DEBUG] Could not re-enable contacts widget: {e}")

    def on_close(self):
        """Properly shuts down the Desktop application and any child windows."""
        print("[DEBUG] Shutting down Desktop application.")
        # Iterate over all child widgets and destroy Toplevel windows (e.g., any Chat windows)
        for widget in self.winfo_children():
            if isinstance(widget, tk.Toplevel):
                try:
                    widget.destroy()
                    print(f"[DEBUG] Destroyed child window: {widget}")
                except Exception as e:
                    print(f"[DEBUG] Error destroying child window: {e}")
        # Finally, destroy the main Desktop window.
        self.destroy()
        print("[DEBUG] Desktop application closed.")
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(
        description="Dungeon Master Desktop with Contacts and Chat Windows")
    parser.add_argument("--hf_key", type=str, required=True, help="Hugging Face authentication token")
    parser.add_argument("--personas_folder", type=str, default="./personas", help="Path to personas folder")
    args = parser.parse_args()

    # Convert the personas_folder to an absolute path relative to the desktop.py file.
    if not os.path.isabs(args.personas_folder):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        args.personas_folder = os.path.join(current_dir, args.personas_folder)
    print("[DEBUG] Starting Desktop with hf_key:", args.hf_key)
    print("[DEBUG] Using personas folder (absolute):", args.personas_folder)

    desktop = Desktop(args.hf_key, args.personas_folder)
    desktop.mainloop()

if __name__ == "__main__":
    main()
