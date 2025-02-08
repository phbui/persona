import tkinter as tk
from tkinter import ttk
import os
import glob
import json
import argparse

# Import your Chat class.
# (Ensure that Chat is a Toplevel subclass rather than a Tk subclass for multi-window management.)
from chat import Chat

class Desktop(tk.Tk):
    def __init__(self, hf_key, personas_folder):
        super().__init__()
        self.title("Dungeon Master Desktop")
        self.geometry("400x600")
        self.hf_key = hf_key
        self.personas_folder = personas_folder

        print("[DEBUG] Desktop initialized with hf_key:", hf_key)
        print("[DEBUG] Using personas folder:", self.personas_folder)

        # Instead of a Listbox, use a Treeview for better control over item appearance.
        self.style = ttk.Style()
        # Increase row height for vertical padding (adjust the value as needed)
        self.style.configure("Treeview", rowheight=30)

        self.contacts_tree = ttk.Treeview(self, columns=("Name",), show="headings", selectmode="browse")
        self.contacts_tree.heading("Name", text="Contact")
        self.contacts_tree.column("Name", anchor="center")
        self.contacts_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Load persona files from the personas folder.
        self.contacts = self.load_personas()
        print("[DEBUG] Loaded", len(self.contacts), "contacts.")
        for contact in self.contacts:
            # Add horizontal padding by prepending and appending spaces.
            
            print("[DEBUG] Loaded contact:", contact['name'].strip(), "from file:", contact['file'])
            self.contacts_tree.insert("", tk.END, values=(contact['name'],))

        # Bind double-click event to open a Chat window for the selected contact.
        self.contacts_tree.bind("<Double-Button-1>", self.open_chat_with_contact)

    def load_personas(self):
        """Load all persona JSON files from the personas folder and return a list of contacts."""
        persona_files = glob.glob(os.path.join(self.personas_folder, "*.json"))
        print("[DEBUG] Found", len(persona_files), "persona files in folder.")
        contacts = []
        for pfile in persona_files:
            print("[DEBUG] Loading persona file:", pfile)
            try:
                with open(pfile, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Assume that the persona JSON contains a field "username" or "name".
                name = data.get("username") or data.get("name") or os.path.basename(pfile).split(".")[0]
                contacts.append({
                    "name": name,
                    "file": pfile
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
            print("[DEBUG] Opening Chat window for contact:", contact["name"], "from file:", persona_path)
            try:
                self.contacts_tree.state(["disabled"])
                print("[DEBUG] Contacts widget disabled.")
            except Exception as e:
                print(f"[DEBUG] Could not disable contacts widget: {e}")
            
            # Pass self (Desktop instance) as the parent to the Chat window.
            chat_window = Chat(self, self.hf_key, persona_path, max_tokens=64)
            chat_window.lift()
            self.wait_window(chat_window)
            try:
                print("[DEBUG] Chat window closed. Conversation end data:", chat_window.conversation_end_data)
                print("[DEBUG] Player model end data:", chat_window.player_model_end_data)
            except Exception as e:
                print("[DEBUG] No relevant info from conversation.")

            try:
                self.contacts_tree.state(["!disabled"])
                print("[DEBUG] Contacts widget re-enabled.")
            except Exception as e:
                print(f"[DEBUG] Could not re-enable contacts widget: {e}")

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
