import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from .record_keeper import RecordKeeper
import numpy as np
import matplotlib.pyplot as plt

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

class EpochExpandableFrame(tk.Frame):
    def __init__(self, parent, epoch_index, epoch_data, *args, **kwargs):
        """
        epoch_index: The epoch number.
        epoch_data: A list of Record objects for this epoch.
        """
        super().__init__(parent, *args, **kwargs)
        self.epoch_index = epoch_index
        self.epoch_data = epoch_data
        self.expanded = False

        self.header_button = tk.Button(
            self,
            text=f"Epoch {epoch_index} (click to expand)",
            relief="raised",
            bg="#4CAF50",
            fg="white",
            font=("Helvetica", 12, "bold"),
            command=self.toggle
        )
        self.header_button.pack(fill="x", padx=5, pady=5)

        # Frame to hold the detailed log; initially, content is hidden.
        self.content_frame = tk.Frame(self)

    def toggle(self):
        if self.expanded:
            self.content_frame.pack_forget()
            self.expanded = False
            self.header_button.config(text=f"Epoch {self.epoch_index} (click to expand)")
        else:
            self.content_frame.pack(fill="both", expand=True, padx=5, pady=5)
            self.expanded = True
            self.header_button.config(text=f"Epoch {self.epoch_index} (click to collapse)")
            self.populate_content()

    def populate_content(self):
        # Clear any existing content.
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        # Create a Text widget to show the epoch's log.
        text_widget = tk.Text(self.content_frame, wrap="word", font=("Helvetica", 12), height=10)
        text_widget.pack(fill="both", expand=True)
        # For each record in this epoch, iterate over its turns.
        for record in self.epoch_data:
            for turn in record.records:
                # Use the record's persona_name and the turn's input_message for display.
                line = f"[{record.persona_name}]: {turn.input_message}\n"
                text_widget.insert("end", line)
        text_widget.config(state="disabled")

class EpochAnalysisPanel(ttk.Frame):
    def __init__(self, master, persona):
        """
        Panel to plot reward over turns (across all epochs) for a given persona.
        """
        super().__init__(master)
        self.persona = persona
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        self.update_button = tk.Button(
            self,
            text="Refresh Analysis",
            command=self.update_plot,
            bg="#4CAF50", fg="white",
            font=("Helvetica", 12, "bold")
        )
        self.update_button.pack(side="bottom", pady=5)
        
        self.update_plot()
                
    def update_plot(self):
            record_keeper = RecordKeeper.instance()
            # Combine all turns for this persona across all epochs.
            all_turns = []
            for epoch in record_keeper.epochs:
                for record in epoch:
                    if record.persona_name == self.persona:
                        all_turns.extend(record.records)

            # Define reward labels and extractor function.
            labels = ["Mental Change Reward", "Notes Reward", "Response Reward", "Response Emotion Reward"]
            def reward_extractor(turn):
                return [
                    getattr(turn, "reward_mental_change", 0),
                    getattr(turn, "notes_reward", 0),
                    getattr(turn, "response_reward", 0),
                    getattr(turn, "response_emotion_reward", 0)
                ]

            ax = self.figure.gca()
            ax.cla()  # Clear the axes.
            if all_turns:
                turn_nums = np.arange(1, len(all_turns) + 1)
                # Prepare data series for each reward type.
                series = [[] for _ in labels]
                for turn in all_turns:
                    rewards = reward_extractor(turn)
                    for i, r in enumerate(rewards):
                        series[i].append(r)
                # Use a colormap to assign colors.
                colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
                for i, data in enumerate(series):
                    ax.plot(turn_nums, data, marker="o", linestyle="-", label=labels[i], color=colors[i])
                ax.set_title(f"Rewards Over Turns for {self.persona}")
                ax.set_xlabel("Turn Number")
                ax.set_ylabel("Reward Value")
                ax.legend()
            else:
                ax.text(0.5, 0.5, "No reward data available.", ha="center", va="center")
            self.canvas.draw()

class EpochRecordKeeperUI:
    def __init__(self, master):
        self.master = master
        self._closing = False
        self.master.title("Epoch Record Keeper")
        self.master.geometry("800x1000")
        
        # Main Notebook with two tabs: Log and Analysis.
        self.main_notebook = ttk.Notebook(master)
        self.main_notebook.pack(fill="both", expand=True)
        
        # LOG TAB: Aggregated log over all epochs as expandable sections.
        self.log_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.log_frame, text="Log")
        self.refresh_log_button = tk.Button(
            self.log_frame,
            text="Refresh Log",
            command=self.refresh_log_overview,
            bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold")
        )
        self.refresh_log_button.pack(side="bottom", pady=10)
        
        # Create a ScrollableFrame to hold epoch expandable frames.
        self.log_scrollable = ScrollableFrame(self.log_frame)
        self.log_scrollable.pack(fill="both", expand=True)
        
        # ANALYZE TAB: Contains a Notebook with sub-tabs for each persona.
        self.analysis_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.analysis_frame, text="Analysis")
        self.analysis_notebook = ttk.Notebook(self.analysis_frame)
        self.analysis_notebook.pack(fill="both", expand=True)
        
        # Create the initial persona tabs.
        self.analysis_panels = {}
        self.update_analysis_tabs()
        
        # Add a button to refresh the persona tabs.
        self.refresh_persona_button = tk.Button(
            self.analysis_frame,
            text="Refresh Persona Tabs",
            command=self.update_analysis_tabs,
            bg="#2196F3", fg="white",
            font=("Helvetica", 12, "bold")
        )
        self.refresh_persona_button.pack(side="bottom", pady=10)
        
        self.main_notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)
        self.master.bind("<Destroy>", self.on_destroy)
    
    def refresh_log_overview(self):
        record_keeper = RecordKeeper.instance()
        # Clear the scrollable frame content.
        for widget in self.log_scrollable.scrollable_frame.winfo_children():
            widget.destroy()
        if not record_keeper.epochs:
            tk.Label(self.log_scrollable.scrollable_frame, text="No epoch data available.", font=("Helvetica", 12)).pack()
            return
        for epoch_idx, epoch in enumerate(record_keeper.epochs, start=1):
            epoch_frame = EpochExpandableFrame(self.log_scrollable.scrollable_frame, epoch_idx, epoch)
            epoch_frame.pack(fill="x", pady=5)
    
    def update_analysis_tabs(self):
        """
        Recheck the epochs for unique personas and add new persona tabs if needed.
        """
        record_keeper = RecordKeeper.instance()
        unique_personas = set()
        for epoch in record_keeper.epochs:
            for record in epoch:
                persona = record.persona_name
                if persona:
                    unique_personas.add(persona)
        if not unique_personas:
            unique_personas = {"Default"}
        
        # Create new tabs for any new personas.
        for persona in sorted(unique_personas):
            if persona not in self.analysis_panels:
                frame = ttk.Frame(self.analysis_notebook)
                self.analysis_notebook.add(frame, text=persona)
                panel = EpochAnalysisPanel(frame, persona)
                panel.pack(fill="both", expand=True)
                self.analysis_panels[persona] = panel
            else:
                # Optionally, refresh the existing panel.
                self.analysis_panels[persona].update_plot()
    
    def on_tab_changed(self, event):
        # Optional: collapse expanded sections when switching tabs.
        pass
    
    def on_close(self):
        if self._closing:
            return
        self._closing = True
        print("[DEBUG] Closing Epoch Record Keeper UI...")
        self.master.destroy()
    
    def on_destroy(self, event):
        if event.widget == self.master and not self._closing:
            self.on_close()
