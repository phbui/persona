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

# This widget shows a single field (key/value) and lets you click to expand or collapse the full value.
class ExpandableField(tk.Frame):
    def __init__(self, master, field_name, field_value, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.field_name = field_name
        self.field_value = field_value
        self.expanded = False

        # Create a truncated summary (if necessary)
        summary = str(field_value)
        if len(summary) > 50:
            summary = summary[:50] + "..."
        self.header = tk.Button(
            self, 
            text=f"{field_name}: {summary}", 
            relief="flat", 
            bg="#ccc", 
            anchor="w", 
            command=self.toggle
        )
        self.header.pack(fill="x")

        # Create the details label that will show the full value.
        self.details = tk.Label(
            self, 
            text=str(field_value), 
            anchor="w", 
            justify="left", 
            bg="#eee", 
            wraplength=700
        )
        # Initially hidden.
        self.details.pack(fill="x", padx=20, pady=2)
        self.details.pack_forget()

    def toggle(self):
        if self.expanded:
            self.details.pack_forget()
            self.expanded = False
        else:
            self.details.pack(fill="x", padx=20, pady=2)
            self.expanded = True

# The TurnFrame widget represents one Turn with an overall header that shows a summary
# and a details frame that contains one ExpandableField widget per field.
class TurnFrame(tk.Frame):
    def __init__(self, master, turn, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.turn = turn
        self.expanded = False
        
        # Create a header button with a summary of the turn.
        summary_input = turn.input_message if len(turn.input_message) < 50 else turn.input_message[:50] + "..."
        summary_response = turn.response if len(turn.response) < 50 else turn.response[:50] + "..."
        summary_text = f"Input: {summary_input} | Response: {summary_response}"
        self.header = tk.Button(
            self, 
            text=summary_text, 
            relief="flat", 
            bg="#ddd", 
            anchor="w", 
            command=self.toggle
        )
        self.header.pack(fill="x")
        
        # Create a details frame that will hold expandable fields.
        self.details_frame = tk.Frame(self, bg="#eee")
        self.details_frame.pack(fill="x", padx=10, pady=5)
        self.details_frame.pack_forget()
        
        # Dynamically get all attributes from the turn object.
        # This uses turn.__dict__ to iterate over all key/value pairs.
        # If needed, you can filter out any keys here.
        for key, value in turn.__dict__.items():
            # Skip internal attributes if necessary (e.g., those starting with "_")
            if key.startswith("_"):
                continue
            # Create an ExpandableField for each attribute.
            ef = ExpandableField(self.details_frame, key, value, bg="#F5F5F5")
            ef.pack(fill="x", padx=5, pady=2)
            
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
