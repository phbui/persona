import argparse
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
from dm_engine import LLM, Conversation, Persona, PlayerModel

class PlayerModelVisualizationGUI(tk.Toplevel):
    def __init__(self, master, player_model: PlayerModel):
        super().__init__(master)
        self.title("Player Model Visualization")
        self.geometry("1000x800")
        self.player_model = player_model

        # Create a Notebook for different visualizations.
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create frames for each visualization.
        self.sentiment_frame = ttk.Frame(self.notebook)
        self.heatmap_frame = ttk.Frame(self.notebook)
        self.decision_flow_frame = ttk.Frame(self.notebook)
        self.embedding_proj_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.sentiment_frame, text="Sentiment & Emotion")
        self.notebook.add(self.heatmap_frame, text="Behavior Heatmap")
        self.notebook.add(self.decision_flow_frame, text="Decision Flow")
        self.notebook.add(self.embedding_proj_frame, text="Embedding Projection")

        # Set up matplotlib figures for each frame.
        self.fig_sentiment, self.ax_sentiment = plt.subplots(figsize=(5, 4))
        self.canvas_sentiment = FigureCanvasTkAgg(self.fig_sentiment, master=self.sentiment_frame)
        self.canvas_sentiment.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig_heatmap, self.ax_heatmap = plt.subplots(figsize=(5, 4))
        self.canvas_heatmap = FigureCanvasTkAgg(self.fig_heatmap, master=self.heatmap_frame)
        self.canvas_heatmap.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig_decision, self.ax_decision = plt.subplots(figsize=(5, 4))
        self.canvas_decision = FigureCanvasTkAgg(self.fig_decision, master=self.decision_flow_frame)
        self.canvas_decision.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig_embed, self.ax_embed = plt.subplots(figsize=(5, 4))
        self.canvas_embed = FigureCanvasTkAgg(self.fig_embed, master=self.embedding_proj_frame)
        self.canvas_embed.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Schedule regular updates
        self.update_visualizations()

    def update_visualizations(self):
        """Update all visualizations."""
        self.update_sentiment_graph()
        self.update_heatmap()
        self.update_decision_flow()
        self.update_embedding_projection()
        # Re-schedule update every 2 seconds
        self.after(2000, self.update_visualizations)

    def update_sentiment_graph(self):
        """Plot a bar chart of sentiment counts."""
        sentiment_counts = self.player_model.get_sentiment_history()
        sentiments = list(sentiment_counts.keys())
        counts = list(sentiment_counts.values())
        self.ax_sentiment.clear()
        self.ax_sentiment.bar(sentiments, counts, color=['green', 'blue', 'red'])
        self.ax_sentiment.set_title("Sentiment Distribution")
        self.ax_sentiment.set_ylabel("Count")
        self.canvas_sentiment.draw()

    def update_heatmap(self):
        """Plot a heatmap of behavior by role vs sentiment."""
        roles, sentiments, heatmap = self.player_model.get_behavior_heatmap_data()
        self.ax_heatmap.clear()
        cax = self.ax_heatmap.imshow(heatmap, cmap="viridis")
        self.ax_heatmap.set_xticks(np.arange(len(sentiments)))
        self.ax_heatmap.set_xticklabels(sentiments)
        self.ax_heatmap.set_yticks(np.arange(len(roles)))
        self.ax_heatmap.set_yticklabels(roles)
        self.ax_heatmap.set_title("Behavior Heatmap")
        self.fig_heatmap.colorbar(cax, ax=self.ax_heatmap)
        self.canvas_heatmap.draw()

    def update_decision_flow(self):
        """Plot the decision flow graph using networkx."""
        self.ax_decision.clear()
        G = self.player_model.get_decision_flow_graph()
        pos = nx.spring_layout(G)
        labels = nx.get_node_attributes(G, "label")
        nx.draw(G, pos, ax=self.ax_decision, with_labels=True, labels=labels, node_color="lightblue", edge_color="gray")
        self.ax_decision.set_title("Decision Flow Graph")
        self.canvas_decision.draw()

    def update_embedding_projection(self):
        """Plot a 2D projection of the player embeddings."""
        self.ax_embed.clear()
        embeddings_2d, roles = self.player_model.get_embedding_projection()
        if embeddings_2d.size:
            colors = ["blue" if role == "user" else "orange" for role in roles]
            self.ax_embed.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.7)
        self.ax_embed.set_title("Player Embedding Projection (PCA)")
        self.canvas_embed.draw()

class ChatGUI(tk.Tk):
    def __init__(self, hf_key, persona_path, max_tokens=32):
        super().__init__()
        self.title("Dungeon Master Engine Chat")
        self.geometry("800x600")
        self.max_tokens = max_tokens

        # Instantiate DM engine classes.
        self.persona = Persona(persona_path)
        self.llm = LLM(secret_key=hf_key, model_name="mistralai/Mistral-7B-Instruct-v0.3")
        self.conversation = Conversation(self.llm, persona=self.persona)

        # Create the player model (hybrid: probabilistic + embedding)
        self.player_model = PlayerModel(embedding_dim=50)

        # Create chat area.
        self.chat_area = ScrolledText(self, wrap=tk.WORD, font=("Helvetica", 16))
        self.chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.chat_area.config(state=tk.DISABLED)
        self.chat_area.tag_configure("user", foreground="white", background="#0B93F6",
                                     justify="right", spacing1=5, spacing3=10, lmargin1=50, rmargin=10)
        self.chat_area.tag_configure("assistant", foreground="black", background="#E5E5EA",
                                     justify="left", spacing1=5, spacing3=10, lmargin1=10, rmargin=50)
        self.chat_area.tag_configure("system", foreground="gray", justify="center", spacing1=5, spacing3=10)

        # Input frame for message entry and send button.
        input_frame = tk.Frame(self)
        input_frame.pack(padx=10, pady=10, fill=tk.X)
        self.input_field = tk.Entry(input_frame, font=("Helvetica", 20))
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        send_button = tk.Button(input_frame, text="Send", font=("Helvetica", 20), command=self.send_message)
        send_button.pack(side=tk.LEFT)

        # Button to open the player model visualization window.
        vis_button = tk.Button(self, text="Show Player Model", font=("Helvetica", 16),
                               command=self.open_player_model_visualization)
        vis_button.pack(pady=(0, 10))

        # Bind Return key to sending message.
        self.bind("<Return>", lambda event: self.send_message())

        # Insert initial system messages.
        self.insert_message("system", f"{self.persona.username} joined the chat.")
        self.insert_message("system", "User joined the chat.")

    def insert_message(self, tag, message):
        """Insert a message into the chat area."""
        self.chat_area.config(state=tk.NORMAL)
        if tag == "user":
            self.chat_area.insert(tk.END, f"You: {message}\n", "user")
        elif tag == "assistant":
            self.chat_area.insert(tk.END, f"{self.persona.username}: {message}\n", "assistant")
        else:
            self.chat_area.insert(tk.END, f"{message}\n", "system")
        self.chat_area.config(state=tk.DISABLED)
        self.chat_area.see(tk.END)
        print(f"[DEBUG] Inserted {tag} message: {message}", flush=True)

    def send_message(self):
        user_message = self.input_field.get().strip()
        if not user_message:
            return
        print(f"[DEBUG] User message: {user_message}", flush=True)
        self.insert_message("user", user_message)
        self.input_field.delete(0, tk.END)

        # Update the player model with the user input.
        self.player_model.update(user_message, role="user")

        # Insert typing indicator for the assistant.
        self.chat_area.config(state=tk.NORMAL)
        typing_text = f"{self.persona.username} is typing..."
        self.chat_area.insert(tk.END, f"{typing_text}\n", "assistant")
        self.chat_area.config(state=tk.DISABLED)
        self.chat_area.see(tk.END)
        print(f"[DEBUG] Inserted typing indicator.", flush=True)

        # Launch background thread to generate assistant response.
        threading.Thread(target=self.get_response, args=(user_message,), daemon=True).start()

    def get_response(self, user_message):
        # Before generating a response, let the player model predict the next assistant response.
        predicted_embedding = self.player_model.predict_next_response()

        try:
            print(f"[DEBUG] Generating response for: {user_message}", flush=True)
            # Call the conversation; this should generate the actual assistant response.
            prompt, assistant_response = self.conversation.chat(user_message, max_new_tokens=self.max_tokens)
        except Exception as e:
            assistant_response = "Error generating response."
            print(f"[ERROR] Exception in get_response: {e}", flush=True)

        # Update the player model with the assistant's response.
        self.player_model.update(assistant_response, role="assistant")
        # Compute a dummy reward based on the similarity between prediction and actual response.
        reward = self.player_model.compute_reward(assistant_response)
        print(f"[DEBUG] Computed reward: {reward:.3f}", flush=True)

        # Replace the typing indicator with the actual response.
        self.after(0, lambda: self.update_response(assistant_response))

    def update_response(self, response):
        self.chat_area.config(state=tk.NORMAL)
        # Find the typing indicator line.
        typing_index = self.chat_area.search(f"{self.persona.username} is typing...", "1.0", tk.END)
        if typing_index:
            line_number = typing_index.split('.')[0]
            start_index = f"{line_number}.0"
            end_index = f"{line_number}.end"
            self.chat_area.delete(start_index, end_index)
            self.chat_area.insert(start_index, f"{self.persona.username}: {response}\n", "assistant")
            print(f"[DEBUG] Replaced typing indicator with response: {response}", flush=True)
        else:
            self.chat_area.insert(tk.END, f"{self.persona.username}: {response}\n", "assistant")
            print(f"[DEBUG] Appended response: {response}", flush=True)
        self.chat_area.config(state=tk.DISABLED)
        self.chat_area.see(tk.END)

    def open_player_model_visualization(self):
        """Launch a new window to visualize the player model."""
        PlayerModelVisualizationGUI(self, self.player_model)

def main():
    parser = argparse.ArgumentParser(description="Dungeon Master Engine Chat GUI with Player Model Visualization")
    parser.add_argument("--hf_key", type=str, required=True, help="Hugging Face authentication token")
    parser.add_argument("--persona_path", type=str, required=True, help="Path to persona JSON file")
    parser.add_argument("--max_tokens", type=int, default=128, help="Default maximum tokens per generation")
    args = parser.parse_args()

    app = ChatGUI(args.hf_key, args.persona_path, args.max_tokens)
    app.mainloop()


if __name__ == "__main__":
    main()
