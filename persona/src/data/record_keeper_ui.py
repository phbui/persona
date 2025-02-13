import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
import matplotlib
import pandas as pd
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
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
    def __init__(self, master, title, description, update_callback, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.update_callback = update_callback
        self.expanded = False
        self.header = tk.Button(self, text=f"{title} - {description}", relief="raised", bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"), command=self.toggle)
        self.header.pack(fill="x")
        self.graph_container = tk.Frame(self)
        self.graph_container.pack(fill="both", expand=True)
        self.graph_container.pack_forget()
        self.figure = None
        self.canvas = None
        self.toolbar = None
        self.data_points = []
        self.detail_windows = []
        self.nav_window = None
        self.nav_index = None
        self.cb = None

    def toggle(self):
        if self.expanded:
            self.collapse()
        else:
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
        for win in self.detail_windows:
            try:
                win.destroy()
            except:
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
                self.open_navigation_window(i)
                self.zoom_to_point(point["x"], point["y"])
                break

    def zoom_to_point(self, x, y, delta=1.0):
        ax = self.figure.gca()
        ax.set_xlim(x - delta, x + delta)
        ax.set_ylim(y - delta, y + delta)
        self.canvas.draw()

    def open_navigation_window(self, index):
        if self.nav_window is not None:
            self.nav_window.destroy()
        self.nav_index = index
        self.nav_window = tk.Toplevel(self.master)
        self.nav_window.title("Turn Details Navigation")
        self.nav_text = scrolledtext.ScrolledText(self.nav_window, wrap=tk.WORD, font=("Helvetica", 10))
        self.nav_text.pack(fill="both", expand=True)
        btn_frame = tk.Frame(self.nav_window)
        btn_frame.pack(fill="x")
        prev_btn = tk.Button(btn_frame, text="Prev", command=self.nav_prev)
        prev_btn.pack(side="left", padx=10, pady=5)
        next_btn = tk.Button(btn_frame, text="Next", command=self.nav_next)
        next_btn.pack(side="right", padx=10, pady=5)
        self.update_nav_window()

    def update_nav_window(self):
        point = self.data_points[self.nav_index]
        turn = point["turn"]
        reward_type = point.get("reward_type", "")
        text = f"Reward Type: {reward_type}\n\n" + str(turn)
        self.nav_text.delete("1.0", tk.END)
        self.nav_text.insert(tk.END, text)

    def nav_prev(self):
        if self.nav_index > 0:
            self.nav_index -= 1
            self.update_nav_window()
            point = self.data_points[self.nav_index]
            self.zoom_to_point(point["x"], point["y"])

    def nav_next(self):
        if self.nav_index < len(self.data_points) - 1:
            self.nav_index += 1
            self.update_nav_window()
            point = self.data_points[self.nav_index]
            self.zoom_to_point(point["x"], point["y"])

class AnalysisUI:
    def __init__(self, master):
        self.master = master
        self.container = ttk.Frame(master)
        self.container.pack(fill="both", expand=True)
        self.graph_panels = []
        self.reward_panel = GraphPanel(self.container, "Aggregate Rewards over Turns", "Line graph for all 4 rewards", self.update_reward_graph)
        self.reward_panel.pack(fill="x", pady=5)
        self.graph_panels.append(self.reward_panel)
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
        ax.set_title("Aggregate Rewards over Turns")
        ax.set_xlabel("Turn Number")
        ax.set_ylabel("Reward Value")
        record_keeper = RecordKeeper.instance()
        data_points = []
        colors = {"reward_mental_change": "blue", "focus_reward": "green", "response_reward": "red", "response_emotion_reward": "purple"}
        for record in record_keeper.records:
            if not record.records:
                continue
            turn_nums = np.arange(1, len(record.records) + 1)
            rm_rewards, f_rewards, r_rewards, re_rewards = [], [], [], []
            for i, turn in enumerate(record.records):
                try:
                    rm = float(turn.reward_mental_change)
                except Exception:
                    try:
                        rm = float(turn.reward_mental_change[0])
                    except Exception:
                        rm = 0
                try:
                    f = float(turn.focus_reward)
                except Exception:
                    try:
                        f = float(turn.focus_reward[0])
                    except Exception:
                        f = 0
                try:
                    r = float(turn.response_reward)
                except Exception:
                    try:
                        r = float(turn.response_reward[0])
                    except Exception:
                        r = 0
                try:
                    re = float(turn.response_emotion_reward)
                except Exception:
                    try:
                        re = float(turn.response_emotion_reward[0])
                    except Exception:
                        re = 0
                rm_rewards.append(rm)
                f_rewards.append(f)
                r_rewards.append(r)
                re_rewards.append(re)
                data_points.append({"x": turn_nums[i], "y": rm, "reward_type": "reward_mental_change", "turn": turn})
                data_points.append({"x": turn_nums[i], "y": f, "reward_type": "focus_reward", "turn": turn})
                data_points.append({"x": turn_nums[i], "y": r, "reward_type": "response_reward", "turn": turn})
                data_points.append({"x": turn_nums[i], "y": re, "reward_type": "response_emotion_reward", "turn": turn})
            ax.plot(turn_nums, rm_rewards, marker="o", color=colors["reward_mental_change"], label=f"{record.persona_name} mental change")
            ax.plot(turn_nums, f_rewards, marker="o", color=colors["focus_reward"], label=f"{record.persona_name} focus")
            ax.plot(turn_nums, r_rewards, marker="o", color=colors["response_reward"], label=f"{record.persona_name} response")
            ax.plot(turn_nums, re_rewards, marker="o", color=colors["response_emotion_reward"], label=f"{record.persona_name} response emotion")
        ax.legend()
        self.reward_panel.data_points = data_points
        figure.canvas.draw()

class RecordKeeperUI:
    def __init__(self, master):
        self.master = master
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
        self.update_log_button = tk.Button(self.log_frame, text="Update Records", command=self.refresh_log, bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"))
        self.update_log_button.pack(side="bottom", pady=10)
        self.analysis_ui = AnalysisUI(self.analyze_frame)

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
            for panel in self.analysis_ui.graph_panels:
                if panel.expanded:
                    panel.collapse()
