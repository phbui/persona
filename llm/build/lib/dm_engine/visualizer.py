import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Visualizer(tk.Toplevel):
    def __init__(self, master, player_model):
        super().__init__(master)
        self.title("Sentiment Scores Over Order")
        self.geometry("800x600")  # Adjust as needed
        self.player_model = player_model

        # Override close protocol for cleanup
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Create a Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Connect zoom and pan events
        self.ax.callbacks.connect('xlim_changed', self.on_x_change)  # Detect x-axis changes
        self.canvas.mpl_connect("scroll_event", self.on_scroll)  # Zoom with mouse scroll
        self.canvas.mpl_connect("button_press_event", self.on_click)  # Click to pan
        self.canvas.mpl_connect("motion_notify_event", self.on_drag)  # Drag to pan
        self.canvas.mpl_connect("key_press_event", self.on_key)  # Reset view with 'r'

        # Variables for dragging (panning)
        self.press_x = None
        self.xlim = None

        # Immediately update visualization
        self.update_visualization()

    def update_visualization(self):
        """Updates the graph with sentiment scores over order."""
        history = self.player_model.get_history_for_visualization()

        # Extract data for plotting
        orders_user = []
        scores_user = []
        orders_llm = []
        scores_llm = []

        for entry in history:
            order = entry["order"]
            compound_score = entry["sentiment_scores"]["compound"]  # Get sentiment score
            role = entry["role"]

            if role in ["user", "player"]:  # Player/User
                orders_user.append(order)
                scores_user.append(compound_score)
            elif role in ["assistant", "llm"]:  # AI/LLM
                orders_llm.append(order)
                scores_llm.append(compound_score)

        # Clear previous graph
        self.ax.clear()

        # Plot user and LLM sentiment lines
        if orders_user:
            self.ax.plot(orders_user, scores_user, marker='o', linestyle='-', color='blue', label="User")
        if orders_llm:
            self.ax.plot(orders_llm, scores_llm, marker='s', linestyle='-', color='orange', label="LLM")

        # Set labels and title
        self.ax.set_title("Sentiment Scores Over Dialogue Order", fontsize=12)
        self.ax.set_xlabel("Dialogue Order")
        self.ax.set_ylabel("Sentiment Score (Compound)")
        self.ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)  # Neutral line
        self.ax.legend()

        # Set initial zoom limits (keep the full view on first render)
        if orders_user or orders_llm:
            all_orders = orders_user + orders_llm
            self.ax.set_xlim(min(all_orders) - 1, max(all_orders) + 1)

        # Refresh canvas
        self.canvas.draw()

    def on_scroll(self, event):
        """Handles zooming in/out with the scroll wheel."""
        x_min, x_max = self.ax.get_xlim()
        x_range = x_max - x_min
        zoom_factor = 0.8 if event.step > 0 else 1.25  # Scroll up zooms in, down zooms out

        # Compute new limits
        new_x_min = event.xdata - (event.xdata - x_min) * zoom_factor
        new_x_max = event.xdata + (x_max - event.xdata) * zoom_factor

        self.ax.set_xlim(new_x_min, new_x_max)
        self.canvas.draw()

    def on_click(self, event):
        """Handles mouse click to start dragging (panning)."""
        if event.button == 1:  # Left mouse button
            self.press_x = event.xdata
            self.xlim = self.ax.get_xlim()

    def on_drag(self, event):
        """Handles dragging (panning) by moving the graph left/right."""
        if self.press_x is None or event.xdata is None:
            return

        dx = self.press_x - event.xdata  # Difference in x position
        x_min, x_max = self.xlim
        self.ax.set_xlim(x_min + dx, x_max + dx)
        self.canvas.draw()

    def on_key(self, event):
        """Resets zoom/pan when 'r' is pressed."""
        if event.key == "r":
            self.update_visualization()

    def on_x_change(self, event):
        """Detects when x-axis changes (for debugging)."""
        x_min, x_max = self.ax.get_xlim()
        print(f"New X Limits: {x_min} to {x_max}")

    def on_close(self):
        """Handles closing the visualization window."""
        self.destroy()
