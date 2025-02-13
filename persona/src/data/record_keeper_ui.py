import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
import datetime
import json

# (Assuming RecordKeeper is defined elsewhere)
from .record_keeper import RecordKeeper

def reduce_to_3d(df):
    pca = PCA(n_components=3)
    return pca.fit_transform(df.values)

class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, background="#F5F5F5")
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

class ExpandableField(tk.Frame):
    def __init__(self, master, field_name, field_value, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.field_name = field_name
        self.field_value = field_value
        self.expanded = False
        summary = str(field_value)
        if len(summary) > 50:
            summary = summary[:50] + "..."
        self.header = tk.Button(self, text=f"{field_name}: {summary}", relief="flat", bg="#ccc", anchor="w", command=self.toggle)
        self.header.pack(fill="x")
        self.details = tk.Label(self, text=str(field_value), anchor="w", justify="left", bg="#eee", wraplength=700)
        self.details.pack(fill="x", padx=20, pady=2)
        self.details.pack_forget()

    def toggle(self):
        if self.expanded:
            self.details.pack_forget()
            self.expanded = False
        else:
            self.details.pack(fill="x", padx=20, pady=2)
            self.expanded = True

class TurnFrame(tk.Frame):
    def __init__(self, master, turn, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.turn = turn
        self.expanded = False
        summary_input = turn.input_message if len(turn.input_message) < 50 else turn.input_message[:50] + "..."
        summary_response = turn.response if len(turn.response) < 50 else turn.response[:50] + "..."
        summary_text = f"Input: {summary_input} | Response: {summary_response}"
        self.header = tk.Button(self, text=summary_text, relief="flat", bg="#ddd", anchor="w", command=self.toggle)
        self.header.pack(fill="x")
        self.details_frame = tk.Frame(self, bg="#eee")
        self.details_frame.pack(fill="x", padx=10, pady=5)
        self.details_frame.pack_forget()
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
    def __init__(self, master, title, description, update_callback, info_callback=None, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.panel_title = title  # Save title for file naming later
        self.update_callback = update_callback
        self.info_callback = info_callback  # Callback for showing/clearing the turn info panel
        self.expanded = False
        self.header = tk.Button(self, text=f"{title} - {description}", relief="raised", bg="#4CAF50", fg="white",
                                font=("Helvetica", 12, "bold"), command=self.toggle)
        self.header.pack(fill="x")
        self.graph_container = tk.Frame(self)
        self.graph_container.pack(fill="both", expand=True)
        self.graph_container.pack_forget()
        self.figure = None
        self.canvas = None
        self.toolbar = None
        self.data_points = []  # Expect dictionaries with keys "x", "y", and "turn"
        self.detail_windows = []

    def toggle(self):
        if self.expanded:
            self.collapse()
        else:
            # Collapse any sibling GraphPanel that is expanded
            for sibling in self.master.winfo_children():
                if isinstance(sibling, GraphPanel) and sibling is not self and sibling.expanded:
                    sibling.collapse()
            self.expand()

    def expand(self):
        self.expanded = True
        self.graph_container.pack(fill="both", expand=True)
        if self.figure is None:
            self.figure = Figure(figsize=(5, 5), dpi=100)
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
        # When collapsing, clear any info panel that might be visible.
        if self.info_callback:
            self.info_callback(None)
        for win in self.detail_windows:
            try:
                win.destroy()
            except Exception:
                pass
        self.detail_windows.clear()

    def on_click(self, event):
        if event.inaxes is None:
            return
        tolerance = 0.5
        for i, point in enumerate(self.data_points):
            dx = event.xdata - point["x"]
            dy = event.ydata - point["y"]
            if np.sqrt(dx*dx + dy*dy) < tolerance:
                if self.info_callback:
                    # Instead of opening a Toplevel window, call the info callback
                    self.info_callback(point["turn"])
                break

class AnalysisUI:
    def __init__(self, master):
        self.master = master
        # Container for all graph panels
        self.container = ttk.Frame(master)
        self.container.pack(fill="both", expand=True)
        # This will hold our turn info panel when needed.
        self.info_frame = None
        self.graph_panels = []
        
        # Pass self.info_callback as the callback for each GraphPanel.
        self.reward_panel = GraphPanel(self.container, 
                                       "Aggregate Rewards over Turns", 
                                       "Line graph for all 4 rewards", 
                                       self.update_reward_graph, 
                                       info_callback=self.info_callback)
        self.reward_panel.pack(fill="x", pady=5)
        self.graph_panels.append(self.reward_panel)
        
        self.mental_state_panel = GraphPanel(self.container, 
                                             "Mental State Evolution", 
                                             "Line graph for mental state attributes", 
                                             self.update_mental_state_graph,
                                             info_callback=self.info_callback)
        self.mental_state_panel.pack(fill="x", pady=5)
        self.graph_panels.append(self.mental_state_panel)
        
        self.input_emotions_panel = GraphPanel(self.container, 
                                               "Input Emotions Over Turns", 
                                               "Line graph for input emotions", 
                                               lambda fig: self.update_emotions_graph(fig, "input"),
                                               info_callback=self.info_callback)
        self.input_emotions_panel.pack(fill="x", pady=5)
        self.graph_panels.append(self.input_emotions_panel)
        
        self.response_emotions_panel = GraphPanel(self.container, 
                                                  "Response Emotions Over Turns", 
                                                  "Line graph for response emotions", 
                                                  lambda fig: self.update_emotions_graph(fig, "response"),
                                                  info_callback=self.info_callback)
        self.response_emotions_panel.pack(fill="x", pady=5)
        self.graph_panels.append(self.response_emotions_panel)
        
        self.update_button = tk.Button(master, text="Update Analysis", command=self.update_all, bg="#4CAF50", fg="white",
                                       font=("Helvetica", 12, "bold"))
        self.update_button.pack(side="bottom", pady=10)

    def info_callback(self, turn):
        """
        Called by a GraphPanel when a turn is clicked (or cleared).
        If turn is None, clear the info panel; otherwise, show it.
        """
        if turn is None:
            self.clear_info_tab()
        else:
            self.show_info_tab(turn)

    def show_info_tab(self, turn):
        # Clear any previous info panel first.
        self.clear_info_tab()
        # Create an info panel at the bottom of the Analyze tab.
        self.info_frame = tk.Frame(self.master, bg="#eee", bd=2, relief="sunken")
        self.info_frame.pack(fill="x", padx=5, pady=5)
        st = scrolledtext.ScrolledText(self.info_frame, wrap=tk.WORD, font=("Helvetica", 10), height=10)
        st.pack(fill="both", expand=True)
        st.insert(tk.END, str(turn))
        st.config(state="disabled")

    def clear_info_tab(self):
        if self.info_frame is not None:
            self.info_frame.destroy()
            self.info_frame = None

    def update_all(self):
        for panel in self.graph_panels:
            if panel.expanded:
                panel.update_callback(panel.figure)
                panel.canvas.draw()

    def update_reward_graph(self, figure):
        self.plot_multi_line(figure, 
                             "Rewards Over Turns", 
                             "Turn Number", 
                             "Reward Value", 
                             ["Mental Change Reward", "Notes Reward", "Response Reward", "Response Emotion Reward"],
                             lambda turn: [turn.reward_mental_change, turn.notes_reward, turn.response_reward, turn.response_emotion_reward])

    def update_mental_state_graph(self, figure):
        self.plot_multi_line(figure, 
                             "Mental State Evolution Over Turns", 
                             "Turn Number", 
                             "Mental State Value", 
                             ["Valence", "Arousal", "Dominance", "Confidence", "Anxiety", "Guilt"],
                             lambda turn: [turn.mental_change["valence"], turn.mental_change["arousal"], turn.mental_change["dominance"], turn.mental_change["confidence"], turn.mental_change["anxiety"], turn.mental_change["guilt"]])

    def update_emotions_graph(self, figure, emotion_type):
        labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
        self.plot_multi_line(
            figure,
            f"{emotion_type.capitalize()} Emotions Over Turns",
            "Turn Number",
            "Emotion Score",
            labels,
            lambda turn: [
                turn.input_message_emotion.get(label, 0) if emotion_type == "input" 
                else turn.response_emotion.get(label, 0) 
                for label in labels
            ]
        )

    def plot_multi_line(self, figure, title, xlabel, ylabel, labels, data_extractor):
        ax = figure.gca()
        ax.cla()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        record_keeper = RecordKeeper.instance()
        colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
        for record in record_keeper.records:
            if not record.records:
                continue
            turn_nums = np.arange(1, len(record.records) + 1)
            series_data = [[] for _ in labels]
            for turn in record.records:
                extracted_data = data_extractor(turn)
                for i, value in enumerate(extracted_data):
                    series_data[i].append(value)
            for i, series in enumerate(series_data):
                ax.plot(turn_nums, series, marker="o", linestyle="-", label=f"{labels[i]}", color=colors[i])
        ax.legend()
        figure.canvas.draw()

class RecordKeeperUI:
    def __init__(self, master):
        self.master = master
        self._closing = False
        self.master.title("Record Keeper")
        self.master.geometry("800x800")
        self.main_notebook = ttk.Notebook(master)
        self.main_notebook.pack(fill="both", expand=True)
        self.log_frame = ttk.Frame(self.main_notebook)
        self.analyze_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.log_frame, text="Log")
        self.main_notebook.add(self.analyze_frame, text="Analyze")
        self.main_notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        self.log_notebook = ttk.Notebook(self.log_frame)
        self.log_notebook.pack(fill="both", expand=True)
        self.tabs = {}
        self.update_log_tabs()
        self.update_log_button = tk.Button(self.log_frame, text="Update Records", command=self.refresh_log, 
                                           bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"))
        self.update_log_button.pack(side="bottom", pady=10)
        self.analysis_ui = AnalysisUI(self.analyze_frame)
        
        # Bind the window close event so we can save the plots.
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)        
        self.master.bind("<Destroy>", self.on_destroy)

    def update_log_tabs(self):
        record_keeper = RecordKeeper.instance()
        for record in record_keeper.records:
            if record.persona_name not in self.tabs:
                frame = ttk.Frame(self.log_notebook)
                self.log_notebook.add(frame, text=record.persona_name)
                scrollable = ScrollableFrame(frame)
                scrollable.pack(fill="both", expand=True)
                self.tabs[record.persona_name] = scrollable.scrollable_frame

    def refresh_log(self):
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

    def on_tab_changed(self, event):
        selected = event.widget.tab(event.widget.index("current"), "text")
        if selected != "Analyze":
            # Collapse any expanded graphs and clear the info panel
            for panel in self.analysis_ui.graph_panels:
                if panel.expanded:
                    panel.collapse()
            self.analysis_ui.clear_info_tab()

    def on_close(self):
        """Save graphs and records in structured folders when closing."""
        record_keeper = RecordKeeper.instance()

        print("[DEBUG] Saving records...")

        # Timestamped session folder
        date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        chat_folder_name = f"chat - {date_str}"

        # Base directory where all saves will go
        current_dir = os.path.dirname(os.path.abspath(__file__))
        saved_dir = os.path.join(current_dir, "saved")
        chat_folder_path = os.path.join(saved_dir, chat_folder_name)

        os.makedirs(chat_folder_path, exist_ok=True)

        for record in record_keeper.records:
            persona_name = record.persona_name or "Unknown"

            # Persona-specific folder inside the chat folder
            persona_folder_path = os.path.join(chat_folder_path, f"{persona_name} - {date_str}")
            os.makedirs(persona_folder_path, exist_ok=True)

            # Save each graph panel's figure in this persona's folder
            for panel in self.analysis_ui.graph_panels:
                if panel.figure:
                    file_name = f"{panel.panel_title}.png"
                    file_path = os.path.join(persona_folder_path, file_name)
                    panel.figure.savefig(file_path)

            # Save the record as a JSON file
            json_file_path = os.path.join(persona_folder_path, f"{persona_name}.json")
            with open(json_file_path, "w", encoding="utf-8") as json_file:
                json.dump(record.to_dict(), json_file, indent=4, ensure_ascii=False)

        print(f"[INFO] Records saved at {chat_folder_path}")
        
        self.master.destroy()

    def on_destroy(self, event):
        # If the main window is being destroyed (via any means) and cleanup hasn't run, call on_close.
        if event.widget == self.master and not self._closing:
            self.on_close()