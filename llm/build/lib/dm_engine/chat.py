import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import threading
import torch
import gc
import queue
import numpy as np
from visualizer import Visualizer
from dm_engine import LLM, Conversation, PlayerModel
from sentence_transformers import SentenceTransformer

class Chat(tk.Toplevel):
    def __init__(self, parent, hf_key, persona, max_tokens=32):
        super().__init__(parent)
        self.title("")
        self.geometry("800x600")
        self.max_tokens = max_tokens
        print("[DEBUG] Initializing ChatGUI.")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.visualization_window = None
        self.persona = persona
        self.llm = LLM(secret_key=hf_key, model_name="mistralai/Mistral-7B-Instruct-v0.3")
        self.conversation = Conversation(self.llm, persona=self.persona)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.player_model = PlayerModel()
        self.chat_area = ScrolledText(self, wrap=tk.WORD, font=("Helvetica", 16))
        self.chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.chat_area.config(state=tk.DISABLED)
        self.chat_area.tag_configure("user", foreground="white", background="#0B93F6",
                                     justify="right", spacing1=5, spacing3=10, lmargin1=50, rmargin=10)
        self.chat_area.tag_configure("assistant", foreground="black", background="#E5E5EA",
                                     justify="left", spacing1=5, spacing3=10, lmargin1=10, rmargin=50)
        self.chat_area.tag_configure("system", foreground="gray", justify="center",
                                     spacing1=5, spacing3=10)
        input_frame = tk.Frame(self)
        input_frame.pack(padx=10, pady=10, fill=tk.X)
        self.input_field = tk.Entry(input_frame, font=("Helvetica", 20))
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        send_button = tk.Button(input_frame, text="Send", font=("Helvetica", 20),
                                command=self.send_message)
        send_button.pack(side=tk.LEFT)
        vis_button = tk.Button(self, text="Show Player Model", font=("Helvetica", 16),
                               command=self.open_player_model_visualization)
        vis_button.pack(pady=(0, 10))
        self.bind("<Return>", lambda event: self.send_message())
        self.insert_message("system", f"{self.persona.username} joined the chat.")
        self.insert_message("system", "User joined the chat.")
        self.response_queue = queue.Queue()
        self.check_response_queue()

    def on_close(self):
        print("[DEBUG] Closing ChatGUI application.")
        try:
            self.conversation_end_data = self.conversation.end_conversation()
            self.player_model_end_data = self.player_model.history
        except Exception as e:
            print(f"[DEBUG] Error ending conversation: {e}")
        if self.visualization_window is not None:
            print("[DEBUG] Closing visualization window.")
            self.visualization_window.destroy()
        print("[DEBUG] Shutting down transformer model.")
        try:
            del self.llm.model
        except Exception as e:
            print(f"[DEBUG] Error deleting model: {e}")
        gc.collect()
        torch.cuda.empty_cache()
        print("[DEBUG] Exiting application.")
        self.destroy()

    def insert_message(self, tag, message):
        self.chat_area.config(state=tk.NORMAL)
        if tag == "user":
            self.chat_area.insert(tk.END, f"You: {message}\n", "user")
        elif tag == "assistant":
            self.chat_area.insert(tk.END, f"{self.persona.username}: {message}\n", "assistant")
        else:
            self.chat_area.insert(tk.END, f"{message}\n", "system")
        self.chat_area.config(state=tk.DISABLED)
        self.chat_area.see(tk.END)
        print(f"[DEBUG] Inserted {tag} message: {message}")

    def get_embedding(self, text):
        emb = self.encoder.encode(text)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    def send_message(self):
        user_message = self.input_field.get().strip()
        if not user_message:
            print("[DEBUG] Empty user message; ignoring.")
            return
        print(f"[DEBUG] User message: {user_message}")
        self.insert_message("user", user_message)
        self.input_field.delete(0, tk.END)
        embedding = self.get_embedding(user_message)
        self.player_model.update(embedding, user_message, role="user")
        print("[DEBUG] Updated player model with user message.")
        self.chat_area.config(state=tk.NORMAL)
        self.after(1000, self.insert_typing_indicator)
        threading.Thread(target=self.get_response, args=(embedding, user_message), daemon=True).start()
        self.update_visualization_if_open()

    def insert_typing_indicator(self):
        typing_text = f"{self.persona.username} is typing..."
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, f"{typing_text}\n", "assistant")
        self.chat_area.config(state=tk.DISABLED)
        self.chat_area.see(tk.END)
        print("[DEBUG] Inserted typing indicator.")

    def get_response(self, embedding, user_message):
        print(f"[DEBUG] Starting response generation thread for message: {user_message}")
        self.persona.check_triggers(embedding)
        try:
            print(f"[DEBUG] Generating response for: {user_message}")
            prompt, assistant_response = self.conversation.chat(user_message, max_new_tokens=self.max_tokens)
        except Exception as e:
            assistant_response = "Error generating response."
            print(f"[ERROR] Exception in get_response: {e}")
        assistant_embedding = self.get_embedding(assistant_response)
        self.player_model.update(assistant_embedding, assistant_response, role="assistant")
        print("[DEBUG] Updated player model with assistant response.")
        self.response_queue.put(assistant_response)

    def check_response_queue(self):
        if not self.winfo_exists():
            return
        try:
            while True:
                response = self.response_queue.get_nowait()
                self.update_response(response)
        except queue.Empty:
            pass
        self.after(100, self.check_response_queue)

    def update_response(self, response):
        self.chat_area.config(state=tk.NORMAL)
        typing_index = self.chat_area.search(f"{self.persona.username} is typing...", "1.0", tk.END)
        if typing_index:
            line_number = typing_index.split('.')[0]
            start_index = f"{line_number}.0"
            end_index = f"{line_number}.end"
            self.chat_area.delete(start_index, end_index)
            self.chat_area.insert(start_index, f"{self.persona.username}: {response}\n", "assistant")
            print(f"[DEBUG] Replaced typing indicator with response: {response}")
        else:
            self.chat_area.insert(tk.END, f"{self.persona.username}: {response}\n", "assistant")
            print(f"[DEBUG] Appended response: {response}")
        self.chat_area.config(state=tk.DISABLED)
        self.chat_area.see(tk.END)
        self.update_visualization_if_open()

    def update_visualization_if_open(self):
        if self.visualization_window is not None:
            print("[DEBUG] Triggering visualization update due to new message.")
            self.visualization_window.update_visualization()

    def open_player_model_visualization(self):
        if self.visualization_window is None or not tk.Toplevel.winfo_exists(self.visualization_window):
            print("[DEBUG] Opening Player Model Visualization window.")
            self.visualization_window = Visualizer(self, self.player_model, self.persona)
        else:
            print("[DEBUG] Player Model Visualization window already open.")
