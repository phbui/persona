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

class GraphPanel(tk.Frame):
    def __init__(self, master, title, description, update_callback, *args, **kwargs):
        """
        update_callback: a function that accepts a Matplotlib Figure and populates it with data.
        """
        super().__init__(master, *args, **kwargs)
        self.update_callback = update_callback
        self.expanded = False
        
        # Header button with title and description.
        self.header = tk.Button(
            self, 
            text=f"{title} - {description}", 
            relief="raised", 
            bg="#4CAF50", 
            fg="white", 
            font=("Helvetica", 12, "bold"), 
            command=self.toggle
        )
        self.header.pack(fill="x")
        
        # Container for the graph; initially hidden.
        self.graph_container = tk.Frame(self)
        self.graph_container.pack(fill="both", expand=True)
        self.graph_container.pack_forget()
        
        self.figure = None
        self.canvas = None
        self.toolbar = None
        self.data_points = []  # List to store metadata for each plotted point.
        self.detail_windows = []  # List to keep track of open detail windows.

    def toggle(self):
        if self.expanded:
            self.collapse()
        else:
            # Collapse all sibling GraphPanels.
            for sibling in self.master.winfo_children():
                if isinstance(sibling, GraphPanel) and sibling is not self and sibling.expanded:
                    sibling.collapse()
            self.expand()

    def expand(self):
        self.expanded = True
        self.graph_container.pack(fill="both", expand=True)
        # Create the figure and canvas if not already created.
        if self.figure is None:
            self.figure = Figure(figsize=(6, 6), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.figure, master=self.graph_container)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.graph_container)
            self.toolbar.update()
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
            self.canvas.mpl_connect("button_press_event", self.on_click)
        self.update_callback(self.figure)
        self.canvas.draw()

    def collapse(self):
        self.expanded = False
        self.graph_container.pack_forget()
        # Close any open detail windows.
        for win in self.detail_windows:
            try:
                win.destroy()
            except:
                pass
        self.detail_windows.clear()

    def on_click(self, event):
        # Only proceed if a point was clicked.
        if event.inaxes is None:
            return
        tolerance = 0.5  # Adjust as needed.
        for point in self.data_points:
            dx = event.xdata - point["x"]
            dy = event.ydata - point["y"]
            if np.sqrt(dx*dx + dy*dy) < tolerance:
                self.open_detail_window(point["turn"])
                break

    def open_detail_window(self, turn):
        # Create a small window with details about the turn.
        detail_win = tk.Toplevel(self.master)
        detail_win.title("Turn Details")
        st = scrolledtext.ScrolledText(detail_win, wrap=tk.WORD, font=("Helvetica", 10))
        st.pack(fill="both", expand=True)
        st.insert(tk.END, str(turn))
        self.detail_windows.append(detail_win)

class AnalysisUI:
    def __init__(self, master):
        self.master = master
        # Create a container for the analysis panels.
        self.container = ttk.Frame(master)
        self.container.pack(fill="both", expand=True)
        self.graph_panels = []

        # Create GraphPanel for Response Reward over Turns.
        self.reward_panel = GraphPanel(
            self.container, 
            "Response Reward over Turns", 
            "Line plot showing response rewards per turn", 
            self.update_reward_graph
        )
        self.reward_panel.pack(fill="x", pady=5)
        self.graph_panels.append(self.reward_panel)

        # Create GraphPanel for Mental State vs Response Emotion.
        self.correlation_panel = GraphPanel(
            self.container, 
            "Mental State vs Response Emotion", 
            "Scatter plot of mental state vs. response emotion", 
            self.update_correlation_graph
        )
        self.correlation_panel.pack(fill="x", pady=5)
        self.graph_panels.append(self.correlation_panel)

        # Create GraphPanel for Focus Trend over Turns.
        self.focus_panel = GraphPanel(
            self.container, 
            "Focus Trend over Turns", 
            "Line plot showing focus per turn", 
            self.update_focus_graph
        )
        self.focus_panel.pack(fill="x", pady=5)
        self.graph_panels.append(self.focus_panel)

        # Create GraphPanel for Distribution of Mental Change.
        self.mental_change_panel = GraphPanel(
            self.container, 
            "Distribution of Mental Change", 
            "Histogram of mental change values", 
            self.update_mental_change_graph
        )
        self.mental_change_panel.pack(fill="x", pady=5)
        self.graph_panels.append(self.mental_change_panel)

        # Create an Update Analysis button.
        self.update_button = tk.Button(master, text="Update Analysis", command=self.update_all, bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"))
        self.update_button.pack(side="bottom", pady=10)

    def update_all(self):
        for panel in self.graph_panels:
            if panel.expanded:
                panel.update_callback(panel.figure)
                panel.canvas.draw()

    def update_reward_graph(self, figure):
        ax = figure.gca()
        ax.cla()
        ax.set_title("Response Reward over Turns")
        ax.set_xlabel("Turn Number")
        ax.set_ylabel("Response Reward")
        data_points = []
        record_keeper = RecordKeeper.instance()
        for record in record_keeper.records:
            if not record.records:
                continue
            turn_nums = np.arange(1, len(record.records) + 1)
            rewards = []
            for i, turn in enumerate(record.records):
                try:
                    reward = float(turn.response_reward)
                except Exception:
                    try:
                        reward = float(turn.response_reward[0])
                    except Exception:
                        reward = 0
                rewards.append(reward)
                data_points.append({"x": turn_nums[i], "y": reward, "turn": turn})
            ax.plot(turn_nums, rewards, marker="o", label=record.persona_name)
        ax.legend()
        # Store data points for click detection.
        self.reward_panel.data_points = data_points

    def update_correlation_graph(self, figure):
        ax = figure.gca()
        ax.cla()
        ax.set_title("Mental State vs Response Emotion")
        ax.set_xlabel("Mental State (After Change)")
        ax.set_ylabel("Response Emotion")
        data_points = []
        record_keeper = RecordKeeper.instance()
        for record in record_keeper.records:
            if not record.records:
                continue
            for turn in record.records:
                try:
                    ms = float(turn.mental_change)
                except Exception:
                    try:
                        ms = float(turn.mental_change[0])
                    except Exception:
                        ms = 0
                try:
                    re_em = float(turn.response_emotion)
                except Exception:
                    try:
                        re_em = float(turn.response_emotion[0])
                    except Exception:
                        re_em = 0
                ax.scatter(ms, re_em, label=record.persona_name)
                data_points.append({"x": ms, "y": re_em, "turn": turn})
        ax.legend()
        self.correlation_panel.data_points = data_points

    def update_focus_graph(self, figure):
        ax = figure.gca()
        ax.cla()
        ax.set_title("Focus Trend over Turns")
        ax.set_xlabel("Turn Number")
        ax.set_ylabel("Focus")
        data_points = []
        record_keeper = RecordKeeper.instance()
        for record in record_keeper.records:
            if not record.records:
                continue
            turn_nums = np.arange(1, len(record.records) + 1)
            focus_values = []
            for i, turn in enumerate(record.records):
                try:
                    focus_val = float(turn.focus)
                except Exception:
                    try:
                        focus_val = float(turn.focus[0])
                    except Exception:
                        focus_val = 0
                focus_values.append(focus_val)
                data_points.append({"x": turn_nums[i], "y": focus_val, "turn": turn})
            ax.plot(turn_nums, focus_values, marker="o", label=record.persona_name)
        ax.legend()
        self.focus_panel.data_points = data_points

    def update_mental_change_graph(self, figure):
        ax = figure.gca()
        ax.cla()
        ax.set_title("Distribution of Mental Change")
        ax.set_xlabel("Mental Change")
        ax.set_ylabel("Frequency")
        data_points = []
        record_keeper = RecordKeeper.instance()
        all_mc = []
        for record in record_keeper.records:
            if not record.records:
                continue
            for turn in record.records:
                try:
                    mc = float(turn.mental_change)
                except Exception:
                    try:
                        mc = float(turn.mental_change[0])
                    except Exception:
                        mc = 0
                all_mc.append(mc)
                data_points.append({"x": mc, "y": None, "turn": turn})
        ax.hist(all_mc, bins=20, alpha=0.5)
        self.mental_change_panel.data_points = data_points

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
        
        # Bind tab change to close any open detail windows.
        self.main_notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        
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
    
    def on_tab_changed(self, event):
        # If the current tab is not Analyze, collapse all AnalysisUI graph panels.
        selected = event.widget.tab(event.widget.index("current"), "text")
        if selected != "Analyze":
            for panel in self.analysis_ui.graph_panels:
                if panel.expanded:
                    panel.collapse()
