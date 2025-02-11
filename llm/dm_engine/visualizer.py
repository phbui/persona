import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import json
import pandas as pd
import seaborn as sns

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
        self.sent_fig = plt.figure(figsize=(7, 8))
        self.compound_ax = self.sent_fig.add_subplot(211)
        self.spectrum_ax = self.sent_fig.add_subplot(212)
        self.sent_canvas = FigureCanvasTkAgg(self.sent_fig, master=self.sentiment_frame)
        self.sent_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.sent_fig.subplots_adjust(hspace=0.4)
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
        self.radar_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.slider_frame = tk.Frame(self.mental_state_frame)
        self.slider_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        self.sliders = {}
        for cat, value in self.persona.mental_state.items():
            var = tk.IntVar(value=value)
            slider = tk.Scale(self.slider_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                              label=cat.capitalize(), variable=var,
                              command=lambda val, cat=cat: self.on_slider_change(cat, val))
            slider.pack(side=tk.LEFT, padx=5, pady=5, expand=True)
            self.sliders[cat] = slider
        self.emotion_vs_ms_frame = tk.Frame(self.notebook)
        self.notebook.add(self.emotion_vs_ms_frame, text="Emotion vs MS")
        self.emotion_vs_ms_fig = plt.figure(figsize=(7, 5))
        self.emotion_vs_ms_ax = self.emotion_vs_ms_fig.add_subplot(111)
        self.emotion_vs_ms_canvas = FigureCanvasTkAgg(self.emotion_vs_ms_fig, master=self.emotion_vs_ms_frame)
        self.emotion_vs_ms_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.update_sentiment_visualization()
        self.update_embeddings_visualization()
        self.update_mental_state_visualization()
        self.update_emotion_vs_ms_visualization()

    def update_visualization(self):
        self.update_sentiment_visualization()
        self.update_embeddings_visualization()
        self.update_mental_state_visualization()
        self.update_emotion_vs_ms_visualization()

    def update_sentiment_visualization(self):
        history = self.player_model.get_history_for_visualization()
        self.compound_ax.clear()
        self.spectrum_ax.clear()
        orders_user, compound_user = [], []
        orders_llm, compound_llm = [], []
        for entry in history:
            order = entry["order"]
            compound_score = entry["sentiment_scores"]["compound"]
            role = entry["role"]
            if role in ["user", "player"]:
                orders_user.append(order)
                compound_user.append(compound_score)
            elif role in ["assistant", "llm"]:
                orders_llm.append(order)
                compound_llm.append(compound_score)
        if orders_user:
            self.compound_ax.plot(orders_user, compound_user, marker='o', linestyle='-', color='blue', label="User")
        if orders_llm:
            self.compound_ax.plot(orders_llm, compound_llm, marker='s', linestyle='-', color='red', label="LLM")
        self.compound_ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        self.compound_ax.set_xlabel("Dialogue Order")
        self.compound_ax.set_ylabel("Compound Score")
        self.compound_ax.set_title("Compound Emotion Score over Dialogue Order")
        self.compound_ax.legend()
        self.compound_ax.grid(True)
        all_orders = sorted(set(orders_user + orders_llm))
        if all_orders:
            self.compound_ax.set_xticks(all_orders)
        emotions = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
        emotion_to_y = {emo: i for i, emo in enumerate(emotions)}
        x_vals = []
        y_vals = []
        sizes = []
        colors = []
        scale_factor = 300
        for entry in history:
            order = entry["order"]
            scores = entry.get("sentiment_scores", {})
            col = "blue" if order % 2 == 1 else "red"
            for emo in emotions:
                score = scores.get(emo, 0)
                x_vals.append(order)
                y_vals.append(emotion_to_y[emo])
                sizes.append(score * scale_factor)
                colors.append(col)
        self.spectrum_ax.scatter(x_vals, y_vals, s=sizes, c=colors, alpha=0.6)
        self.spectrum_ax.set_xlabel("Dialogue Order")
        self.spectrum_ax.set_ylabel("Emotion")
        self.spectrum_ax.set_title("Emotion Spectrum over Dialogue Order")
        self.spectrum_ax.set_yticks(list(emotion_to_y.values()))
        self.spectrum_ax.set_yticklabels(list(emotion_to_y.keys()), fontsize=8)
        self.spectrum_ax.grid(True)
        if all_orders:
            self.spectrum_ax.set_xticks(all_orders)
        self.sent_canvas.draw_idle()

    def update_embeddings_visualization(self):
        history = self.player_model.get_history_for_visualization()
        self.emb_ax.clear()
        xs, ys, zs, colors = [], [], [], []
        for entry in history:
            emb = entry["embedding"]
            xs.append(emb[0])
            ys.append(emb[1])
            zs.append(emb[2])
            colors.append('blue' if entry["role"] in ["user", "player"] else 'red')
        self.emb_scatter = self.emb_ax.scatter(xs, ys, zs, c=colors, depthshade=True, s=60)
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
                self.emb_ax.quiver(user_x[i], user_y[i], user_z[i], dx, dy, dz,
                                   arrow_length_ratio=0.1, color='blue', linewidth=1)
        if len(assistant_x) > 1:
            self.emb_ax.plot(assistant_x, assistant_y, assistant_z, color='red', linewidth=2, label="Assistant Path")
            for i in range(len(assistant_x) - 1):
                dx = assistant_x[i+1] - assistant_x[i]
                dy = assistant_y[i+1] - assistant_y[i]
                dz = assistant_z[i+1] - assistant_z[i]
                self.emb_ax.quiver(assistant_x[i], assistant_y[i], assistant_z[i], dx, dy, dz,
                                   arrow_length_ratio=0.1, color='red', linewidth=1)
        if hasattr(self.persona, "embedded_triggers") and self.persona.embedded_triggers:
            triggers_embeddings = np.array([trigger["embedding"] for trigger in self.persona.embedded_triggers])
            reduced_triggers = self.player_model.reduce_array(triggers_embeddings)
            self.emb_ax.scatter(reduced_triggers[:, 0], reduced_triggers[:, 1], reduced_triggers[:, 2],
                                c='red', marker='*', s=100, label="Triggers")
        for entry in history:
            if "chunked_embeddings" in entry and entry["chunked_embeddings"] is not None:
                role = entry["role"]
                color = "blue" if role in ["user", "player"] else "red"
                chunk_embs = entry["chunked_embeddings"]
                if chunk_embs.shape[0] > 0:
                    main_emb = entry["embedding"]
                    for i in range(chunk_embs.shape[0]):
                        self.emb_ax.plot([main_emb[0], chunk_embs[i, 0]],
                                         [main_emb[1], chunk_embs[i, 1]],
                                         [main_emb[2], chunk_embs[i, 2]],
                                         linestyle='dotted', color=color, linewidth=1)
                    for i in range(chunk_embs.shape[0]):
                        self.emb_ax.scatter(chunk_embs[i, 0], chunk_embs[i, 1], chunk_embs[i, 2],
                                            c=color, s=10, marker='o')
        self.emb_ax.set_title("3D Embedding Visualization", fontsize=12)
        self.emb_ax.set_xlabel("PC1")
        self.emb_ax.set_ylabel("PC2")
        self.emb_ax.set_zlabel("PC3")
        self.emb_ax.legend()
        self.emb_canvas.draw_idle()

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
        self.radar_ax.set_ylim(-20, 100)
        self.radar_ax.plot(angles, values, linewidth=2, linestyle='solid', marker='o', markersize=10)
        self.radar_ax.fill(angles, values, 'b', alpha=0.25)
        self.radar_canvas.draw_idle()
            
    def update_emotion_vs_ms_visualization(self):
        print("DEBUG: Starting update_emotion_vs_ms_visualization")
        history = self.player_model.get_history_for_visualization()
        print(f"DEBUG: Retrieved history with {len(history)} entries.")
        
        # Filter for non-player entries.
        non_player = [entry for entry in history if entry["role"] in ["assistant", "llm"]]
        print(f"DEBUG: Filtered non-player entries: {len(non_player)} found.")
        print(f"Non-Player history data: {non_player}")
        
        if non_player:
            # Use the full non-player history (remove slider usage)
            subset = non_player
            print(f"DEBUG: Using full non-player history: {len(subset)} entries")
            
            emotions = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
            ms_keys = ["valence", "arousal", "dominance", "confidence", "anxiety", "guilt"]

            rows = []
            for i, entry in enumerate(subset, start=1):
                print(f"DEBUG: Processing dialogue entry {i}")
                emo_dict = {emotion: entry["sentiment_scores"].get(emotion, 0) for emotion in emotions}
                print(f"DEBUG: Emotion dict for dialogue {i}: {emo_dict}")
                ms_dict = {key: entry["mental_state"].get(key, 0) for key in ms_keys}
                print(f"DEBUG: Mental state dict for dialogue {i}: {ms_dict}")
                row = {"dialogue": i}
                row.update(emo_dict)
                row.update(ms_dict)
                print(f"DEBUG: Combined row for dialogue {i}: {row}")
                rows.append(row)
            
            df = pd.DataFrame(rows)
            print(f"DEBUG: DataFrame created with shape {df.shape}")
            cols = emotions + ms_keys
            df = df[cols]
            print("DEBUG: DataFrame columns reordered.")
            
            if len(df) < 2:
                print("DEBUG: Not enough data points to compute correlations.")
                self.emotion_vs_ms_ax.clear()
                self.emotion_vs_ms_ax.text(0.5, 0.5, "Not enough data to compute correlations",
                                            ha="center", va="center")
            else:
                corr_matrix = df.corr()
                print("DEBUG: Correlation matrix calculated:")
                print(corr_matrix)
                self.emotion_vs_ms_ax.clear()
                sns.heatmap(corr_matrix.loc[emotions, ms_keys],
                            annot=True,
                            cmap="coolwarm",
                            ax=self.emotion_vs_ms_ax,
                            cbar_kws={"label": "Correlation"},
                            annot_kws={"fontsize":8})
                self.emotion_vs_ms_ax.set_title("Correlation: Emotions vs Mental State", fontsize=10)
                print("DEBUG: Heatmap plotted on the axes.")
        else:
            print("DEBUG: Not enough non-player data found, displaying message on axes.")
            self.emotion_vs_ms_ax.clear()
            self.emotion_vs_ms_ax.text(0.5, 0.5, "Not enough non-player data", ha="center", va="center")
        
        self.emotion_vs_ms_fig.tight_layout()
        self.emotion_vs_ms_canvas.draw_idle()
        print("DEBUG: Finished update_emotion_vs_ms_visualization and canvas updated.")

    def on_scroll(self, event):
        if event.xdata is None:
            return
        x_min, x_max = self.sent_ax.get_xlim()
        zoom_factor = 0.8 if event.step > 0 else 1.25
        new_x_min = event.xdata - (event.xdata - x_min) * zoom_factor
        new_x_max = event.xdata + (x_max - event.xdata) * zoom_factor
        self.sent_ax.set_xlim(new_x_min, new_x_max)
        self.sent_canvas.draw_idle()

    def on_click(self, event):
        if event.button == 3 and event.xdata is not None:
            self.open_info_window_sentiment(event)
            return
        if event.button == 1 and event.xdata is not None:
            self.press_x = event.xdata
            self.xlim = self.sent_ax.get_xlim()

    def on_drag(self, event):
        if self.press_x is None or event.xdata is None:
            return
        dx = self.press_x - event.xdata
        x_min, x_max = self.xlim
        self.sent_ax.set_xlim(x_min + dx, x_max + dx)
        self.sent_canvas.draw_idle()

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
        self.emb_canvas.draw_idle()

    def on_close(self):
        data = {
            "history": self.player_model.get_history_for_visualization(),
            "mental_state": self.persona.mental_state
        }
        if hasattr(self.persona, "embedded_triggers"):
            data["embedded_triggers"] = self.persona.embedded_triggers
        file_name = "visualization_data.json"
        try:
            with open(file_name, "w") as f:
                json.dump(data, f, indent=4)
            print(f"Data successfully saved to {file_name}")
        except Exception as e:
            print(f"An error occurred while saving data: {e}")
        self.destroy()

    def on_slider_change(self, category, value):
        try:
            new_val = int(value)
        except ValueError:
            new_val = 0
        current_val = self.persona.mental_state.get(category, 0)
        delta = new_val - current_val
        if delta != 0:
            self.persona.update_mental_state({category: delta})
            self.update_mental_state_visualization()
