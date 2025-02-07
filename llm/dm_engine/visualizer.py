import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting (may be implicit in recent matplotlib)
import numpy as np

class Visualizer(tk.Toplevel):
    def __init__(self, master, player_model):
        super().__init__(master)
        self.title("Visualizer: Sentiment & Embeddings")
        self.geometry("900x700")  # Adjust window size as needed
        self.player_model = player_model

        # Override the close protocol for cleanup
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Create a Notebook (tabbed interface)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # ------------------------------
        # Tab 1: Sentiment Graph
        # ------------------------------
        self.sentiment_frame = tk.Frame(self.notebook)
        self.notebook.add(self.sentiment_frame, text="Sentiment")

        # Create a sentiment figure and canvas
        self.sent_fig, self.sent_ax = plt.subplots(figsize=(7, 5))
        self.sent_canvas = FigureCanvasTkAgg(self.sent_fig, master=self.sentiment_frame)
        self.sent_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # (Optional) Connect events for zoom/pan on sentiment tab
        self.sent_ax.callbacks.connect('xlim_changed', self.on_x_change)
        self.sent_canvas.mpl_connect("scroll_event", self.on_scroll)
        self.sent_canvas.mpl_connect("button_press_event", self.on_click)
        self.sent_canvas.mpl_connect("button_release_event", self.on_release)
        self.sent_canvas.mpl_connect("motion_notify_event", self.on_drag)
        self.sent_canvas.mpl_connect("key_press_event", self.on_key)

        # Variables for dragging (panning) in sentiment plot
        self.press_x = None
        self.xlim = None

        # ------------------------------
        # Tab 2: Embeddings 3D Graph
        # ------------------------------
        self.embeddings_frame = tk.Frame(self.notebook)
        self.notebook.add(self.embeddings_frame, text="Embeddings")

        # Create a 3D figure and canvas for embeddings
        self.emb_fig = plt.figure(figsize=(7, 5))
        self.emb_ax = self.emb_fig.add_subplot(111, projection='3d')
        self.emb_canvas = FigureCanvasTkAgg(self.emb_fig, master=self.embeddings_frame)
        self.emb_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Enable interactive rotation in the 3D plot
        self.emb_ax.mouse_init()

        # Immediately update both visualizations
        self.update_sentiment_visualization()
        self.update_embeddings_visualization()

    def update_visualization(self):
        self.update_sentiment_visualization()
        self.update_embeddings_visualization()

    def update_sentiment_visualization(self):
        """Updates the sentiment graph (tab 1) with sentiment scores over dialogue order."""
        history = self.player_model.get_history_for_visualization()

        # Extract data for plotting
        orders_user, scores_user = [], []
        orders_llm, scores_llm = [], []

        for entry in history:
            order = entry["order"]
            compound_score = entry["sentiment_scores"]["compound"]  # Get sentiment score
            role = entry["role"]
            if role in ["user", "player"]:
                orders_user.append(order)
                scores_user.append(compound_score)
            elif role in ["assistant", "llm"]:
                orders_llm.append(order)
                scores_llm.append(compound_score)

        # Clear previous graph
        self.sent_ax.clear()

        # Plot user and LLM sentiment lines
        if orders_user:
            self.sent_ax.plot(orders_user, scores_user, marker='o', linestyle='-', color='blue', label="User")
        if orders_llm:
            self.sent_ax.plot(orders_llm, scores_llm, marker='s', linestyle='-', color='orange', label="LLM")

        # Set labels and title
        self.sent_ax.set_title("Sentiment Scores Over Dialogue Order", fontsize=12)
        self.sent_ax.set_xlabel("Dialogue Order")
        self.sent_ax.set_ylabel("Sentiment Score (Compound)")
        self.sent_ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)  # Neutral line
        self.sent_ax.legend()

        # Set initial zoom limits if available
        if orders_user or orders_llm:
            all_orders = orders_user + orders_llm
            self.sent_ax.set_xlim(min(all_orders) - 1, max(all_orders) + 1)

        self.sent_canvas.draw()

    def update_embeddings_visualization(self):
        """Updates the embeddings 3D graph (tab 2) with reduced 3D embeddings."""
        history = self.player_model.get_history_for_visualization()

        # Clear previous 3D plot
        self.emb_ax.clear()

        # Extract embeddings and roles
        xs, ys, zs = [], [], []
        colors = []
        for entry in history:
            emb = entry["embedding"]  # 3D vector
            xs.append(emb[0])
            ys.append(emb[1])
            zs.append(emb[2])
            # Color-code by role: blue for user/player, orange for assistant/llm.
            if entry["role"] in ["user", "player"]:
                colors.append('blue')
            else:
                colors.append('orange')

        # Create a 3D scatter plot
        self.emb_ax.scatter(xs, ys, zs, c=colors, depthshade=True, s=60)
        self.emb_ax.set_title("3D Embedding Visualization", fontsize=12)
        self.emb_ax.set_xlabel("PC1")
        self.emb_ax.set_ylabel("PC2")
        self.emb_ax.set_zlabel("PC3")

        # Optional: set equal aspect ratio (if desired)
        # This may require additional adjustments.
        
        self.emb_canvas.draw()

    # --- The following event handlers are for the sentiment plot only ---

    def on_scroll(self, event):
        """Handles zooming in/out with the scroll wheel for the sentiment plot."""
        if event.xdata is None:
            return  # Skip if xdata is not defined
        x_min, x_max = self.sent_ax.get_xlim()
        # Choose zoom factor: scroll up (event.step > 0) zooms in, down zooms out
        zoom_factor = 0.8 if event.step > 0 else 1.25
        new_x_min = event.xdata - (event.xdata - x_min) * zoom_factor
        new_x_max = event.xdata + (x_max - event.xdata) * zoom_factor
        self.sent_ax.set_xlim(new_x_min, new_x_max)
        self.sent_canvas.draw()

    def on_click(self, event):
        """Handles mouse click to start dragging (panning) for the sentiment plot."""
        if event.button == 1:  # Left mouse button
            self.press_x = event.xdata
            self.xlim = self.sent_ax.get_xlim()

    def on_drag(self, event):
        """Handles dragging (panning) by moving the sentiment graph left/right."""
        if self.press_x is None or event.xdata is None:
            return
        dx = self.press_x - event.xdata  # Difference in x position
        x_min, x_max = self.xlim
        self.sent_ax.set_xlim(x_min + dx, x_max + dx)
        self.sent_canvas.draw()

    def on_release(self, event):
        """Resets panning variables when the mouse button is released."""
        self.press_x = None
        self.xlim = None

    def on_key(self, event):
        """Resets zoom/pan of the sentiment plot when 'r' is pressed."""
        if event.key == "r":
            self.update_sentiment_visualization()

    def on_x_change(self, event):
        """Detects when x-axis changes (for debugging)."""
        x_min, x_max = self.sent_ax.get_xlim()
        print(f"New X Limits: {x_min} to {x_max}")

    def on_close(self):
        """Handles closing the visualization window."""
        self.destroy()
