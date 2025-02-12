import tkinter as tk
from tkinter import ttk, scrolledtext
from .record_keeper import RecordKeeper

class RecordKeeperUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Record Keeper")
        self.master.geometry("800x600")
        
        # Create a Notebook widget (tabbed interface).
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill="both", expand=True)
        
        # Dictionary to map each record's persona name to its text widget.
        self.tabs = {}
        
        # Initialize tabs for any existing records.
        self.update_tabs()
        
        # Start periodic UI updates.
        self.refresh()

    def update_tabs(self):
        """Check the singleton RecordKeeper for new records and add a new tab if needed."""
        record_keeper = RecordKeeper.instance()  # Assumes you have a RecordKeeper singleton.
        for record in record_keeper.records:
            if record.persona_name not in self.tabs:
                # Create a new frame for the record.
                frame = ttk.Frame(self.notebook)
                self.notebook.add(frame, text=record.persona_name)
                # Create a scrollable text widget to display the record of turns.
                text_widget = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=("Helvetica", 12))
                text_widget.pack(fill="both", expand=True)
                self.tabs[record.persona_name] = text_widget

    def refresh(self):
        """Periodically update each tab with the latest turns for each record."""
        record_keeper = RecordKeeper.instance()
        # Make sure tabs are up to date.
        self.update_tabs()
        
        for record in record_keeper.records:
            text_widget = self.tabs.get(record.persona_name)
            if text_widget:
                # Build a detailed content string by calling __str__ on each Turn.
                content = ""
                for turn in record.records:
                    content += str(turn) + "\n\n"
                text_widget.delete("1.0", tk.END)
                text_widget.insert(tk.END, content)
                text_widget.see(tk.END)  # Scroll to bottom.
        
        # Schedule the next update after 1000 milliseconds.
        self.master.after(1000, self.refresh)