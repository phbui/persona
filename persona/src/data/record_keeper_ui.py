import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

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

# The TurnFrame widget represents one Turn with an overall header and expandable fields.
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
        for key, value in turn.__dict__.items():
            if key.startswith("_"):
                continue
            ef = ExpandableField(self.details_frame, key, value, bg="#F5F5F5")
            ef.pack(fill="x", padx=5, pady=2)
            
    def toggle(self):
        if self.expanded:
            self.details_frame.pack_forget()
            self.expanded = False
        else:
            self.details_frame.pack(fill="x", padx=10, pady=5)
            self.expanded = True

class AnalysisUI:
    def __init__(self, master):
        self.master = master
        # Do not call configure(bg="white") on a ttk widget.
        # Instead, if desired, you can set a style for ttk widgets.
        
        # Create an analysis container.
        self.analysis_container = ttk.Frame(master)
        self.analysis_container.pack(fill="both", expand=True)
        
        # Create a Matplotlib Figure with 2x2 subplots.
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.ax_reward = self.figure.add_subplot(221)      # Response Reward over Turns
        self.ax_correlation = self.figure.add_subplot(222)   # Mental State vs. Response Emotion
        self.ax_focus = self.figure.add_subplot(223)         # Focus Trend over Turns
        self.ax_mental_change = self.figure.add_subplot(224) # Distribution of Mental Change
        
        # Set titles and labels.
        self.ax_reward.set_title("Response Reward over Turns")
        self.ax_reward.set_xlabel("Turn Number")
        self.ax_reward.set_ylabel("Response Reward")
        
        self.ax_correlation.set_title("Mental State vs. Response Emotion")
        self.ax_correlation.set_xlabel("Mental State (After Change)")
        self.ax_correlation.set_ylabel("Response Emotion")
        
        self.ax_focus.set_title("Focus Trend over Turns")
        self.ax_focus.set_xlabel("Turn Number")
        self.ax_focus.set_ylabel("Focus")
        
        self.ax_mental_change.set_title("Distribution of Mental Change")
        self.ax_mental_change.set_xlabel("Mental Change")
        self.ax_mental_change.set_ylabel("Frequency")
        
        # Embed the figure.
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.analysis_container)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.analysis_container)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Create an Update Analysis button.
        self.update_button = tk.Button(master, text="Update Analysis", command=self.update_analysis,
                                       bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"))
        self.update_button.pack(side="bottom", pady=10)
    
    def update_analysis(self):
        """Update analysis plots based on current record data.
           Each Turn is a data point; x-axis is turn number.
        """
        self.ax_reward.cla()
        self.ax_correlation.cla()
        self.ax_focus.cla()
        self.ax_mental_change.cla()
        
        # Reset titles and labels.
        self.ax_reward.set_title("Response Reward over Turns")
        self.ax_reward.set_xlabel("Turn Number")
        self.ax_reward.set_ylabel("Response Reward")
        
        self.ax_correlation.set_title("Mental State vs. Response Emotion")
        self.ax_correlation.set_xlabel("Mental State (After Change)")
        self.ax_correlation.set_ylabel("Response Emotion")
        
        self.ax_focus.set_title("Focus Trend over Turns")
        self.ax_focus.set_xlabel("Turn Number")
        self.ax_focus.set_ylabel("Focus")
        
        self.ax_mental_change.set_title("Distribution of Mental Change")
        self.ax_mental_change.set_xlabel("Mental Change")
        self.ax_mental_change.set_ylabel("Frequency")
        
        record_keeper = RecordKeeper.instance()
        
        # Loop over each record.
        for record in record_keeper.records:
            if not record.records:
                continue
            turn_nums = np.arange(1, len(record.records) + 1)
            rewards = []
            mental_states = []      # Assuming mental_change represents the updated mental state.
            response_emotions = []
            focus_values = []
            mental_changes = []
            
            for turn in record.records:
                try:
                    reward = float(turn.response_reward)
                except Exception:
                    try:
                        reward = float(turn.response_reward[0])
                    except Exception:
                        reward = 0
                rewards.append(reward)
                
                try:
                    ms = float(turn.mental_change)
                except Exception:
                    try:
                        ms = float(turn.mental_change[0])
                    except Exception:
                        ms = 0
                mental_states.append(ms)
                
                try:
                    re_em = float(turn.response_emotion)
                except Exception:
                    try:
                        re_em = float(turn.response_emotion[0])
                    except Exception:
                        re_em = 0
                response_emotions.append(re_em)
                
                try:
                    focus_val = float(turn.focus)
                except Exception:
                    try:
                        focus_val = float(turn.focus[0])
                    except Exception:
                        focus_val = 0
                focus_values.append(focus_val)
                
                try:
                    mc = float(turn.mental_change)
                except Exception:
                    try:
                        mc = float(turn.mental_change[0])
                    except Exception:
                        mc = 0
                mental_changes.append(mc)
            
            self.ax_reward.plot(turn_nums, rewards, label=record.persona_name)
            self.ax_focus.plot(turn_nums, focus_values, label=record.persona_name)
            self.ax_correlation.scatter(mental_states, response_emotions, label=record.persona_name)
            self.ax_mental_change.hist(mental_changes, bins=20, alpha=0.5, label=record.persona_name)
        
        self.ax_reward.legend()
        self.ax_focus.legend()
        self.ax_correlation.legend()
        self.ax_mental_change.legend()
        self.canvas.draw()

class RecordKeeperUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Record Keeper")
        self.master.geometry("800x600")
        
        # Create a top-level Notebook with two main tabs: Log & Analyze.
        self.main_notebook = ttk.Notebook(master)
        self.main_notebook.pack(fill="both", expand=True)
        
        self.log_frame = ttk.Frame(self.main_notebook)
        self.analyze_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.log_frame, text="Log")
        self.main_notebook.add(self.analyze_frame, text="Analyze")
        
        # Create a sub-notebook inside the Log tab for each record.
        self.log_notebook = ttk.Notebook(self.log_frame)
        self.log_notebook.pack(fill="both", expand=True)
        self.tabs = {}  # Maps each record's persona_name to its scrollable frame.
        self.update_log_tabs()
        
        # Create an Update button in the Log tab.
        self.update_log_button = tk.Button(self.log_frame, text="Update Records", command=self.refresh_log, bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"))
        self.update_log_button.pack(side="bottom", pady=10)
        
        # Instantiate the Analysis UI as part of the RecordKeeperUI.
        self.analysis_ui = AnalysisUI(self.analyze_frame)
    
    def update_log_tabs(self):
        """Check the RecordKeeper singleton for new records and add a new tab if needed."""
        record_keeper = RecordKeeper.instance()  # Assumes RecordKeeper singleton exists.
        for record in record_keeper.records:
            if record.persona_name not in self.tabs:
                frame = ttk.Frame(self.log_notebook)
                self.log_notebook.add(frame, text=record.persona_name)
                scrollable = ScrollableFrame(frame)
                scrollable.pack(fill="both", expand=True)
                self.tabs[record.persona_name] = scrollable.scrollable_frame
                # print(f"[DEBUG] Added tab for {record.persona_name}")
    
    def refresh_log(self):
        """Manually update each tab with the latest turns for each record."""
        record_keeper = RecordKeeper.instance()
        self.update_log_tabs()
        for record in record_keeper.records:
            content_frame = self.tabs.get(record.persona_name)
            if content_frame is not None:
                for widget in content_frame.winfo_children():
                    widget.destroy()
                for turn in record.records:
                    tf = TurnFrame(content_frame, turn, bg="#F5F5F5", bd=1, relief="solid")
                    tf.pack(fill="x", padx=5, pady=5)
                # print(f"[DEBUG] Updated records for {record.persona_name}")
