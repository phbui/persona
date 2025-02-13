import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import datetime
import json
from .record_keeper import RecordKeeper

#---------------------------
# Helper UI widget: a scrollable frame
#---------------------------
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

#---------------------------
# EpochAnalysisPanel: displays a single graph for one persona.
#---------------------------
class EpochAnalysisPanel(ttk.Frame):
    def __init__(self, master, persona):
        """
        Panel to plot average reward per epoch for a given persona.
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
        epochs = record_keeper.epochs  # List of epochs; each epoch is a list of turn records.
        epoch_avg_rewards = []
        for epoch in epochs:
            # Filter records for this persona.
            persona_records = [record for record in epoch if record.get("persona_name") == self.persona]
            rewards = [record.get("reward", 0) for record in persona_records]
            if rewards:
                avg_reward = sum(rewards) / len(rewards)
            else:
                avg_reward = 0
            epoch_avg_rewards.append(avg_reward)
        
        ax = self.figure.gca()
        ax.cla()
        if epoch_avg_rewards:
            epochs_range = range(1, len(epoch_avg_rewards) + 1)
            ax.plot(epochs_range, epoch_avg_rewards, marker="o", linestyle="-", color="blue")
            ax.set_title(f"Average Reward for {self.persona}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Average Reward")
        else:
            ax.text(0.5, 0.5, "No epoch data available.", ha="center", va="center")
        self.canvas.draw()

#---------------------------
# EpochRecordKeeperUI: main UI for epoch-level records.
# Contains two main tabs: Log and Analysis.
#---------------------------
class EpochRecordKeeperUI:
    def __init__(self, master):
        self.master = master
        self._closing = False
        self.master.title("Epoch Record Keeper")
        self.master.geometry("800x1000")
        
        # Main Notebook with two tabs: Log and Analysis.
        self.main_notebook = ttk.Notebook(master)
        self.main_notebook.pack(fill="both", expand=True)
        
        # LOG TAB: Aggregated log over all epochs.
        self.log_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.log_frame, text="Log")
        self.log_text = tk.Text(self.log_frame, wrap=tk.WORD, font=("Helvetica", 12))
        self.log_text.pack(fill="both", expand=True)
        self.log_scrollbar = ttk.Scrollbar(self.log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=self.log_scrollbar.set)
        self.log_scrollbar.pack(side="right", fill="y")
        self.refresh_log_button = tk.Button(
            self.log_frame,
            text="Refresh Log",
            command=self.refresh_log_overview,
            bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold")
        )
        self.refresh_log_button.pack(side="bottom", pady=10)
        
        # ANALYZE TAB: Contains a Notebook with sub-tabs for each persona.
        self.analysis_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.analysis_frame, text="Analysis")
        self.analysis_notebook = ttk.Notebook(self.analysis_frame)
        self.analysis_notebook.pack(fill="both", expand=True)
        
        # Determine unique personas from the epochs.
        record_keeper = RecordKeeper.instance()
        unique_personas = set()
        for epoch in record_keeper.epochs:
            for record in epoch:
                persona = record.get("persona_name")
                if persona:
                    unique_personas.add(persona)
        if not unique_personas:
            unique_personas = {"Default"}
        
        self.analysis_panels = {}
        for persona in sorted(unique_personas):
            frame = ttk.Frame(self.analysis_notebook)
            self.analysis_notebook.add(frame, text=persona)
            panel = EpochAnalysisPanel(frame, persona)
            panel.pack(fill="both", expand=True)
            self.analysis_panels[persona] = panel
        
        self.main_notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)
        self.master.bind("<Destroy>", self.on_destroy)
    
    def refresh_log_overview(self):
        """Aggregate logs from all epochs and display in the Log tab."""
        record_keeper = RecordKeeper.instance()
        self.log_text.delete("1.0", tk.END)
        if not record_keeper.epochs:
            self.log_text.insert(tk.END, "No epoch data available.")
            return
        for epoch_idx, epoch in enumerate(record_keeper.epochs, start=1):
            self.log_text.insert(tk.END, f"Epoch {epoch_idx}:\n")
            for turn in epoch:
                # Each turn is expected to be a dict with keys: player_name and message.
                line = f"[{turn.get('player_name', 'Unknown')}]: {turn.get('message', '')}\n"
                self.log_text.insert(tk.END, line)
            self.log_text.insert(tk.END, "\n" + "-" * 40 + "\n\n")
    
    def on_tab_changed(self, event):
        # You can add behavior to collapse/refresh panels when switching tabs if needed.
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
