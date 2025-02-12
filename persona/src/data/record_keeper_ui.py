import tkinter as tk
from tkinter import ttk, scrolledtext
from .record_keeper import RecordKeeper  # Assumes a singleton RecordKeeper exists

# A scrollable frame that will be used in each tab.
class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, background="#F5F5F5")
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

# A custom widget representing a single Turn as an expandable/collapsible frame.
class TurnFrame(tk.Frame):
    def __init__(self, master, turn, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.turn = turn
        self.expanded = False
        
        # Create a header button with a summary of the turn.
        summary_input = turn.input_message if len(turn.input_message) < 50 else turn.input_message[:50] + "..."
        summary_response = turn.response if len(turn.response) < 50 else turn.response[:50] + "..."
        summary_text = f"Input: {summary_input} | Response: {summary_response}"
        self.header = tk.Button(self, text=summary_text, relief="flat", bg="#ddd", anchor="w", command=self.toggle)
        self.header.pack(fill="x")
        
        # Create a details frame that contains labels for each field.
        self.details_frame = tk.Frame(self, bg="#eee")
        # Initially hide the details frame.
        self.details_frame.pack(fill="x", padx=10, pady=5)
        self.details_frame.pack_forget()
        
        # Dictionary of fields to display.
        fields = {
            "input_message": turn.input_message,
            "input_message_embedding": turn.input_message_embedding,
            "input_message_emotion": turn.input_message_emotion,
            "mental_change": turn.mental_change,
            "reward_mental_change": turn.reward_mental_change,
            "focus": turn.focus,
            "focus_reward": turn.focus_reward,
            "prompt": turn.prompt,
            "response": turn.response,
            "response_reward": turn.response_reward,
            "response_emotion": turn.response_emotion,
            "response_emotion_reward": turn.response_emotion_reward,
            "policy": turn.policy
        }
        for key, value in fields.items():
            label = tk.Label(self.details_frame, text=f"{key}: {value}", anchor="w", justify="left", bg="#eee", wraplength=700)
            label.pack(fill="x", anchor="w")
            
    def toggle(self):
        if self.expanded:
            self.details_frame.pack_forget()
            self.expanded = False
        else:
            self.details_frame.pack(fill="x", padx=10, pady=5)
            self.expanded = True

# The main RecordKeeper UI, which creates a tab for each record and updates it manually.
class RecordKeeperUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Record Keeper")
        self.master.geometry("800x600")
        
        # Create a Notebook (tabbed interface).
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill="both", expand=True)
        
        # Dictionary mapping each record's persona_name to its scrollable frame.
        self.tabs = {}
        
        # Initialize tabs for any existing records.
        self.update_tabs()
        
        # Create an Update button at the bottom of the window.
        self.update_button = tk.Button(master, text="Update Records", command=self.refresh, bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"))
        self.update_button.pack(side="bottom", pady=10)
        
    def update_tabs(self):
        """Check the RecordKeeper singleton for new records and add a new tab if needed."""
        record_keeper = RecordKeeper.instance()  # Assumes RecordKeeper singleton exists.
        for record in record_keeper.records:
            if record.persona_name not in self.tabs:
                # Create a new frame for the record.
                frame = ttk.Frame(self.notebook)
                self.notebook.add(frame, text=record.persona_name)
                # Create a scrollable frame inside the tab.
                scrollable = ScrollableFrame(frame)
                scrollable.pack(fill="both", expand=True)
                self.tabs[record.persona_name] = scrollable.scrollable_frame
                # print(f"[DEBUG] Added tab for {record.persona_name}")

    def refresh(self):
        """Manually update each tab with the latest turns for each record."""
        record_keeper = RecordKeeper.instance()
        # Update tabs if there are new records.
        self.update_tabs()
        
        for record in record_keeper.records:
            content_frame = self.tabs.get(record.persona_name)
            if content_frame is not None:
                # Clear previous content.
                for widget in content_frame.winfo_children():
                    widget.destroy()
                # Create a TurnFrame widget for each turn.
                for turn in record.records:
                    tf = TurnFrame(content_frame, turn, bg="#F5F5F5", bd=1, relief="solid")
                    tf.pack(fill="x", padx=5, pady=5)
                # print(f"[DEBUG] Updated records for {record.persona_name}")
