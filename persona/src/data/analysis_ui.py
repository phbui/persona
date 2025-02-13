import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from .record_keeper import RecordKeeper

def reduce_to_3d(df):
    pca = PCA(n_components=3)
    return pca.fit_transform(df.values)

class GraphPanel(tk.Frame):
    def __init__(self, master, title, description, update_callback, info_callback=None, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.panel_title = title 
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
    def __init__(self, master, persona=None):
        self.master = master
        self.persona = persona  # If specified, only analyze records for this persona
        # Container for all graph panels
        self.container = ttk.Frame(master)
        self.container.pack(fill="both", expand=True)
        # This will hold our turn info panel when needed.
        self.info_frame = None
        self.graph_panels = []
        
        # Create graph panels with callbacks (unchanged from before)
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
        
        self.input_vs_mental_panel = GraphPanel(
            self.container,
            "Input vs Mental Correlation Matrix",
            "Heatmap comparing input_message_emotion and mental_change",
            lambda fig: self.plot_dict_correlation_matrix(fig, "input_message_emotion", "mental_change"),
            info_callback=self.info_callback
        )
        self.input_vs_mental_panel.pack(fill="x", pady=5)
        self.graph_panels.append(self.input_vs_mental_panel)
        
        self.mental_vs_response_panel = GraphPanel(
            self.container,
            "Mental vs Response Correlation Matrix",
            "Heatmap comparing mental_change and response_emotion",
            lambda fig: self.plot_dict_correlation_matrix(fig, "mental_change", "response_emotion"),
            info_callback=self.info_callback
        )
        self.mental_vs_response_panel.pack(fill="x", pady=5)
        self.graph_panels.append(self.mental_vs_response_panel)
        
        self.update_button = tk.Button(master, text="Update Analysis", command=self.update_all, bg="#4CAF50", fg="white",
                                       font=("Helvetica", 12, "bold"))
        self.update_button.pack(side="bottom", pady=10)

    def get_filtered_records(self):
        # Return only records matching the specified persona (if any)
        record_keeper = RecordKeeper.instance()
        if self.persona:
            return [record for record in record_keeper.records if record.persona_name == self.persona]
        return record_keeper.records

    def info_callback(self, turn):
        if turn is None:
            self.clear_info_tab()
        else:
            self.show_info_tab(turn)

    def show_info_tab(self, turn):
        self.clear_info_tab()
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
        records = self.get_filtered_records()
        self.plot_multi_line(figure, 
                             "Rewards Over Turns", 
                             "Turn Number", 
                             "Reward Value", 
                             ["Mental Change Reward", "Notes Reward", "Response Reward", "Response Emotion Reward"],
                             lambda turn: [turn.reward_mental_change, turn.notes_reward, turn.response_reward, turn.response_emotion_reward],
                             records)

    def update_mental_state_graph(self, figure):
        records = self.get_filtered_records()
        self.plot_multi_line(figure, 
                             "Mental State Evolution Over Turns", 
                             "Turn Number", 
                             "Mental State Value", 
                             ["Valence", "Arousal", "Dominance", "Confidence", "Anxiety", "Guilt"],
                             lambda turn: [turn.mental_change["valence"], turn.mental_change["arousal"], turn.mental_change["dominance"], turn.mental_change["confidence"], turn.mental_change["anxiety"], turn.mental_change["guilt"]],
                             records)

    def update_emotions_graph(self, figure, emotion_type):
        labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
        records = self.get_filtered_records()
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
            ],
            records)

    def plot_multi_line(self, figure, title, xlabel, ylabel, labels, data_extractor, records):
        ax = figure.gca()
        ax.cla()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
        for record in records:
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

    def plot_dict_correlation_matrix(self, figure, field1, field2, default=0, record_keeper=None):
        records = self.get_filtered_records()
        # Gather all sub-keys from both fields.
        subkeys1 = set()
        subkeys2 = set()
        for record in records:
            for turn in record.records:
                dict1 = getattr(turn, field1, {}) or {}
                dict2 = getattr(turn, field2, {}) or {}
                if isinstance(dict1, dict):
                    subkeys1.update(dict1.keys())
                if isinstance(dict2, dict):
                    subkeys2.update(dict2.keys())
        subkeys1 = sorted(list(subkeys1))
        subkeys2 = sorted(list(subkeys2))

        # Initialize a correlation matrix.
        corr_matrix = np.zeros((len(subkeys1), len(subkeys2)))
        for i, key1 in enumerate(subkeys1):
            for j, key2 in enumerate(subkeys2):
                x_values, y_values = [], []
                for record in records:
                    for turn in record.records:
                        dict1 = getattr(turn, field1, {}) or {}
                        dict2 = getattr(turn, field2, {}) or {}
                        x = dict1.get(key1, default) if isinstance(dict1, dict) else default
                        y = dict2.get(key2, default) if isinstance(dict2, dict) else default
                        x_values.append(x)
                        y_values.append(y)
                if len(x_values) > 1:
                    corr, _ = pearsonr(x_values, y_values)
                else:
                    corr = 0.0
                corr_matrix[i, j] = corr

        # Plot a heatmap in the provided figure.
        figure.clf()
        ax = figure.add_subplot(111)
        cax = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        figure.colorbar(cax)
        ax.set_xticks(np.arange(len(subkeys2)))
        ax.set_yticks(np.arange(len(subkeys1)))
        ax.set_xticklabels(subkeys2, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(subkeys1, fontsize=8)
        ax.set_title(f"{field1} vs {field2} Correlation Matrix", fontsize=12)
        # Annotate each cell with the correlation value.
        for i in range(len(subkeys1)):
            for j in range(len(subkeys2)):
                ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)
        figure.tight_layout()
        figure.canvas.draw()
