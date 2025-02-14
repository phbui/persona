import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
import os
import datetime
import json
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from .record_keeper import RecordKeeper
from .analysis_ui import AnalysisUI

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

class RecordKeeperUI:
    def __init__(self, master):
        self.master = master
        self._closing = False
        self.master.title("Record Keeper")
        self.master.geometry("800x1000")
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

        # Create a Notebook inside the Analyze tab to separate analysis per persona.
        self.analysis_notebook = ttk.Notebook(self.analyze_frame)
        self.analysis_notebook.pack(fill="both", expand=True)
        # For each unique persona in the records, create a sub-tab.
        record_keeper = RecordKeeper.instance()
        unique_personas = sorted({record.persona_name for record in record_keeper.records if record.persona_name})
        # If there are no personas, create one default tab.
        if not unique_personas:
            unique_personas = ["Default"]
        self.analysis_ui_instances = {}
        for persona in unique_personas:
            frame = ttk.Frame(self.analysis_notebook)
            self.analysis_notebook.add(frame, text=persona)
            analysis_ui = AnalysisUI(frame, persona=persona)
            self.analysis_ui_instances[persona] = analysis_ui

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
            # Collapse any expanded graphs in all analysis UI instances
            for analysis_ui in self.analysis_ui_instances.values():
                for panel in analysis_ui.graph_panels:
                    if panel.expanded:
                        panel.collapse()
                analysis_ui.clear_info_tab()

    def on_close(self):
        if self._closing:
            return
        self._closing = True

        record_keeper = RecordKeeper.instance()
        print("[DEBUG] Refreshing analysis before saving records...")

        # Force update for all graph panels in all AnalysisUI instances.
        for analysis_ui in self.analysis_ui_instances.values():
            for panel in analysis_ui.graph_panels:
                # Determine a valid container: try panel.graph_container first, then self.master.
                container = None
                try:
                    if panel.graph_container and panel.graph_container.winfo_exists():
                        container = panel.graph_container
                    elif self.master and self.master.winfo_exists():
                        container = self.master
                except Exception:
                    container = None

                if panel.figure is None:
                    # Create a new figure.
                    panel.figure = Figure(figsize=(5, 5), dpi=100)
                    # Use TkAgg if we have a valid container, otherwise use Agg.
                    if container is not None:
                        panel.canvas = FigureCanvasTkAgg(panel.figure, master=container)
                        panel.canvas.mpl_connect("button_press_event", panel.on_click)
                    else:
                        panel.canvas = FigureCanvasAgg(panel.figure)
                else:
                    # If figure exists but no canvas, create one using a valid container if possible.
                    if not hasattr(panel, 'canvas') or panel.canvas is None:
                        if container is not None:
                            panel.canvas = FigureCanvasTkAgg(panel.figure, master=container)
                            panel.canvas.mpl_connect("button_press_event", panel.on_click)
                        else:
                            panel.canvas = FigureCanvasAgg(panel.figure)

                # Update and draw the figure.
                panel.update_callback(panel.figure)
                panel.canvas.draw()

        try:
            if self.master and self.master.winfo_exists():
                self.master.update_idletasks()  # Process any pending UI updates
        except Exception:
            pass  # If master doesn't exist, ignore.

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

            # Save each graph panel's figure in this persona's folder for every AnalysisUI instance.
            for analysis_ui in self.analysis_ui_instances.values():
                for panel in analysis_ui.graph_panels:
                    if panel.figure:
                        panel.canvas.draw()  # Ensure the figure is drawn before saving
                        file_name = f"{panel.panel_title}.png"
                        file_path = os.path.join(persona_folder_path, file_name)
                        panel.figure.savefig(file_path)
                        print(f"[INFO] Saved plot: {file_path}")

            # Save the record as a JSON file
            json_file_path = os.path.join(persona_folder_path, f"{persona_name}.json")
            with open(json_file_path, "w", encoding="utf-8") as json_file:
                json.dump(record.to_dict(), json_file, indent=4, ensure_ascii=False)
            print(f"[INFO] Saved record: {json_file_path}")

        print(f"[INFO] Records saved at {chat_folder_path}")
        self.master.destroy()

    def on_destroy(self, event):
        # If the main window is being destroyed (via any means) and cleanup hasn't run, call on_close.
        if event.widget == self.master and not self._closing:
            self.on_close()