import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from sklearn.decomposition import PCA

class Visualizer(tk.Toplevel):
    def __init__(self, master, player_model, persona):
        super().__init__(master)
        self.title("Visualizer: Sentiment & Embeddings")
        self.geometry("900x700")
        self.player_model = player_model
        self.persona = persona 
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.sentiment_frame = tk.Frame(self.notebook)
        self.notebook.add(self.sentiment_frame, text="Sentiment")
        self.sent_fig, self.sent_ax = plt.subplots(figsize=(7, 5))
        self.sent_canvas = FigureCanvasTkAgg(self.sent_fig, master=self.sentiment_frame)
        self.sent_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.sent_ax.callbacks.connect('xlim_changed', self.on_x_change)
        self.sent_canvas.mpl_connect("scroll_event", self.on_scroll)
        self.sent_canvas.mpl_connect("button_press_event", self.on_click)
        self.sent_canvas.mpl_connect("button_release_event", self.on_release)
        self.sent_canvas.mpl_connect("motion_notify_event", self.on_drag)
        self.sent_canvas.mpl_connect("key_press_event", self.on_key)
        self.press_x = None
        self.xlim = None

        self.embeddings_frame = tk.Frame(self.notebook)
        self.notebook.add(self.embeddings_frame, text="Embeddings")
        self.emb_fig = plt.figure(figsize=(7, 5))
        self.emb_ax = self.emb_fig.add_subplot(111, projection='3d')
        self.emb_canvas = FigureCanvasTkAgg(self.emb_fig, master=self.embeddings_frame)
        self.emb_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.emb_ax.mouse_init()
        self.emb_canvas.mpl_connect("scroll_event", self.on_scroll_emb)
        self.emb_canvas.mpl_connect("button_press_event", self.on_info_click_emb)

        self.mental_state_frame = tk.Frame(self.notebook)
        self.notebook.add(self.mental_state_frame, text="Mental State")
        self.radar_fig = plt.figure(figsize=(7, 5))
        self.radar_ax = self.radar_fig.add_subplot(111, polar=True)
        self.radar_canvas = FigureCanvasTkAgg(self.radar_fig, master=self.mental_state_frame)
        self.radar_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.update_sentiment_visualization()
        self.update_embeddings_visualization()
        self.update_mental_state_visualization()

    def update_visualization(self):
        self.update_sentiment_visualization()
        self.update_embeddings_visualization()
        self.update_mental_state_visualization()

    def update_sentiment_visualization(self):
        history = self.player_model.get_history_for_visualization()
        orders_user, scores_user, orders_llm, scores_llm = [], [], [], []
        for entry in history:
            order = entry["order"]
            compound_score = entry["sentiment_scores"]["compound"]
            role = entry["role"]
            if role in ["user", "player"]:
                orders_user.append(order)
                scores_user.append(compound_score)
            elif role in ["assistant", "llm"]:
                orders_llm.append(order)
                scores_llm.append(compound_score)
        self.sent_ax.clear()
        if orders_user:
            self.sent_ax.plot(orders_user, scores_user, marker='o', linestyle='-', color='blue', label="User")
        if orders_llm:
            self.sent_ax.plot(orders_llm, scores_llm, marker='s', linestyle='-', color='orange', label="LLM")
        self.sent_ax.set_title("Sentiment Scores Over Dialogue Order", fontsize=12)
        self.sent_ax.set_xlabel("Dialogue Order")
        self.sent_ax.set_ylabel("Sentiment Score (Compound)")
        self.sent_ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        self.sent_ax.legend()
        if orders_user or orders_llm:
            all_orders = orders_user + orders_llm
            self.sent_ax.set_xlim(min(all_orders) - 1, max(all_orders) + 1)
        self.sent_canvas.draw()

    def update_embeddings_visualization(self):
        history = self.player_model.get_history_for_visualization()
        self.emb_ax.clear()
        xs, ys, zs, colors = [], [], [], []
        for entry in history:
            emb = entry["embedding"]
            xs.append(emb[0])
            ys.append(emb[1])
            zs.append(emb[2])
            if entry["role"] in ["user", "player"]:
                colors.append('blue')
            else:
                colors.append('orange')
        self.emb_ax.scatter(xs, ys, zs, c=colors, depthshade=True, s=60)
        sorted_history = sorted(history, key=lambda x: x["order"])
        user_x, user_y, user_z, assistant_x, assistant_y, assistant_z = [], [], [], [], [], []
        for entry in sorted_history:
            emb = entry["embedding"]
            if entry["role"] in ["user", "player"]:
                user_x.append(emb[0])
                user_y.append(emb[1])
                user_z.append(emb[2])
            elif entry["role"] in ["assistant", "llm"]:
                assistant_x.append(emb[0])
                assistant_y.append(emb[1])
                assistant_z.append(emb[2])
        if len(user_x) > 1:
            self.emb_ax.plot(user_x, user_y, user_z, color='blue', linewidth=2, label="User Path")
            for i in range(len(user_x) - 1):
                dx = user_x[i+1] - user_x[i]
                dy = user_y[i+1] - user_y[i]
                dz = user_z[i+1] - user_z[i]
                self.emb_ax.quiver(user_x[i], user_y[i], user_z[i], dx, dy, dz, arrow_length_ratio=0.1, color='blue', linewidth=1)
        if len(assistant_x) > 1:
            self.emb_ax.plot(assistant_x, assistant_y, assistant_z, color='orange', linewidth=2, label="Assistant Path")
            for i in range(len(assistant_x) - 1):
                dx = assistant_x[i+1] - assistant_x[i]
                dy = assistant_y[i+1] - assistant_y[i]
                dz = assistant_z[i+1] - assistant_z[i]
                self.emb_ax.quiver(assistant_x[i], assistant_y[i], assistant_z[i], dx, dy, dz, arrow_length_ratio=0.1, color='orange', linewidth=1)
        if hasattr(self.persona, "embedded_triggers") and self.persona.embedded_triggers:
            triggers_embeddings = np.array([trigger["embedding"] for trigger in self.persona.embedded_triggers])
            if len(triggers_embeddings) == 1:
                reduced_triggers = triggers_embeddings[:, :3]
            elif len(triggers_embeddings) == 2:
                pca = PCA(n_components=2)
                reduced_2d = pca.fit_transform(triggers_embeddings)
                reduced_triggers = np.hstack([reduced_2d, np.zeros((2, 1))])
            else:
                pca = PCA(n_components=3)
                reduced_triggers = pca.fit_transform(triggers_embeddings)
            self.emb_ax.scatter(reduced_triggers[:, 0], reduced_triggers[:, 1], reduced_triggers[:, 2],
                                c='red', marker='*', s=100, label="Triggers")
        # For each history entry, if chunked embeddings exist, plot them.
        for entry in history:
            if "chunked_embeddings" in entry and entry["chunked_embeddings"] is not None:
                # Assume entry["chunked_embeddings"] is a NumPy array of shape (n, 3)
                chunk_embs = entry["chunked_embeddings"]
                if chunk_embs.shape[0] > 0:
                    main_emb = entry["embedding"]
                    # Connect main embedding to the first chunk.
                    self.emb_ax.plot([main_emb[0], chunk_embs[0, 0]],
                                    [main_emb[1], chunk_embs[0, 1]],
                                    [main_emb[2], chunk_embs[0, 2]],
                                    linestyle='dotted', color='blue', linewidth=1)
                if chunk_embs.shape[0] > 1:
                    # Plot each chunk with small markers (no label).
                    for i in range(chunk_embs.shape[0]):
                        self.emb_ax.scatter(chunk_embs[i, 0], chunk_embs[i, 1], chunk_embs[i, 2],
                                            c='blue', s=20, marker='o')
                    # Draw a dotted line connecting all chunk embeddings in order.
                    self.emb_ax.plot(chunk_embs[:, 0], chunk_embs[:, 1], chunk_embs[:, 2],
                                    linestyle='dotted', color='blue', linewidth=1)

        self.emb_ax.set_title("3D Embedding Visualization", fontsize=12)
        self.emb_ax.set_xlabel("PC1")
        self.emb_ax.set_ylabel("PC2")
        self.emb_ax.set_zlabel("PC3")
        self.emb_ax.legend()
        self.emb_canvas.draw()

    def update_mental_state_visualization(self):
        ms = self.persona.mental_state
        categories = list(ms.keys())
        values = list(ms.values())
        values += values[:1]
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        self.radar_ax.clear()
        self.radar_ax.set_theta_offset(np.pi / 2)
        self.radar_ax.set_theta_direction(-1)
        self.radar_ax.set_xticks(angles[:-1])
        self.radar_ax.set_xticklabels(categories)
        self.radar_ax.set_ylim(-2, 10)
        self.radar_ax.plot(angles, values, linewidth=2, linestyle='solid')
        self.radar_ax.fill(angles, values, 'b', alpha=0.25)
        self.radar_canvas.draw()

    def on_scroll(self, event):
        if event.xdata is None: return
        x_min, x_max = self.sent_ax.get_xlim()
        zoom_factor = 0.8 if event.step > 0 else 1.25
        new_x_min = event.xdata - (event.xdata - x_min) * zoom_factor
        new_x_max = event.xdata + (x_max - event.xdata) * zoom_factor
        self.sent_ax.set_xlim(new_x_min, new_x_max)
        self.sent_canvas.draw()

    def on_click(self, event):
        if event.button == 3 and event.xdata is not None:
            self.open_info_window_sentiment(event)
            return
        if event.button == 1 and event.xdata is not None:
            self.press_x = event.xdata
            self.xlim = self.sent_ax.get_xlim()

    def on_drag(self, event):
        if self.press_x is None or event.xdata is None: return
        dx = self.press_x - event.xdata
        x_min, x_max = self.xlim
        self.sent_ax.set_xlim(x_min + dx, x_max + dx)
        self.sent_canvas.draw()

    def on_release(self, event):
        self.press_x = None
        self.xlim = None

    def on_key(self, event):
        if event.key == "r":
            self.update_sentiment_visualization()

    def on_x_change(self, event):
        x_min, x_max = self.sent_ax.get_xlim()
        print(f"New X Limits: {x_min} to {x_max}")

    def open_info_window_sentiment(self, event):
        x_click = event.xdata
        y_click = event.ydata
        history = self.player_model.get_history_for_visualization()
        best_entry = None
        best_dist = float('inf')
        for entry in history:
            x_val = entry["order"]
            y_val = entry["sentiment_scores"]["compound"]
            dist = np.sqrt((x_click - x_val)**2 + (y_click - y_val)**2)
            if dist < best_dist:
                best_dist = dist
                best_entry = entry
        if best_dist < 1.0:
            self.show_info_window(best_entry)

    def on_info_click_emb(self, event):
        if event.button == 3:
            if hasattr(self, 'emb_scatter'):
                contains, attr = self.emb_scatter.contains(event)
                if contains:
                    index = attr["ind"][0]
                    history = self.player_model.get_history_for_visualization()
                    if index < len(history):
                        self.show_info_window(history[index])

    def show_info_window(self, entry):
        info_win = tk.Toplevel(self)
        info_win.title(f"Detail for Message {entry['order']}")
        info_win.geometry("400x300")
        text = tk.Text(info_win, wrap="word", width=50, height=15)
        text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        details = f"Order: {entry['order']}\n"
        details += f"Timestamp: {entry['timestamp']}\n"
        details += f"Role: {entry['role']}\n\n"
        details += f"Message:\n{entry['message']}\n\n"
        details += f"Sentiment: {entry['sentiment']}\n\n"
        details += "Sentiment Scores:\n"
        for key, value in entry["sentiment_scores"].items():
            details += f"  {key}: {value}\n"
        text.insert("1.0", details)
        text.config(state="disabled")
        close_btn = tk.Button(info_win, text="Close", command=info_win.destroy)
        close_btn.pack(pady=5)

    def on_scroll_emb(self, event):
        x_min, x_max = self.emb_ax.get_xlim3d()
        y_min, y_max = self.emb_ax.get_ylim3d()
        z_min, z_max = self.emb_ax.get_zlim3d()
        zoom_factor = 0.8 if event.step > 0 else 1.25
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2
        new_x_range = (x_max - x_min) * zoom_factor / 2
        new_y_range = (y_max - y_min) * zoom_factor / 2
        new_z_range = (z_max - z_min) * zoom_factor / 2
        self.emb_ax.set_xlim3d([x_center - new_x_range, x_center + new_x_range])
        self.emb_ax.set_ylim3d([y_center - new_y_range, y_center + new_y_range])
        self.emb_ax.set_zlim3d([z_center - new_z_range, z_center + new_z_range])
        self.emb_canvas.draw()

    def on_close(self):
        self.destroy()
