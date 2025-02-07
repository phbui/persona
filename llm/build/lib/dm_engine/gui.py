import argparse
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import threading
import re
from dm_engine import LLM, Conversation, Persona

class ChatGUI(tk.Tk):
    def __init__(self, hf_key, persona_path, max_tokens=64):
        super().__init__()
        self.title("Dungeon Master Engine Chat")
        self.geometry("800x600")
        self.max_tokens = max_tokens

        # Instantiate Persona, LLM, and Conversation.
        self.persona = Persona(persona_path)
        self.llm = LLM(secret_key=hf_key, model_name="mistralai/Mistral-7B-Instruct-v0.3")
        self.conversation = Conversation(self.llm, persona=self.persona)

        # Create a ScrolledText widget for the chat area.
        self.chat_area = ScrolledText(self, wrap=tk.WORD, font=("Helvetica", 22))
        self.chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.chat_area.config(state=tk.DISABLED)  # read-only but selectable

        # Configure tags for styling and alignment.
        self.chat_area.tag_configure("user", foreground="white", background="#0B93F6",
                                     justify="right", spacing1=5, spacing3=10, lmargin1=50, rmargin=10)
        self.chat_area.tag_configure("assistant", foreground="black", background="#E5E5EA",
                                     justify="left", spacing1=5, spacing3=10, lmargin1=10, rmargin=50)
        self.chat_area.tag_configure("system", foreground="gray", justify="center", spacing1=5, spacing3=10)

        # Create an input field and Send button at the bottom.
        input_frame = tk.Frame(self)
        input_frame.pack(padx=10, pady=10, fill=tk.X)
        self.input_field = tk.Entry(input_frame, font=("Helvetica", 24))
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        send_button = tk.Button(input_frame, text="Send", font=("Helvetica", 24), command=self.send_message)
        send_button.pack(side=tk.LEFT)

        # Bind the Return key to send the message.
        self.bind("<Return>", lambda event: self.send_message())

        # Insert system join messages.
        self.insert_message("system", f"{self.persona.username} joined the chat.")
        self.insert_message("system", "User joined the chat.")

    def insert_message(self, tag, message):
        """Insert a message into the chat area using the specified tag."""
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
        # Insert the user's message.
        self.insert_message("user", user_message)
        self.input_field.delete(0, tk.END)

        # Insert a "typing..." indicator for the assistant.
        self.chat_area.config(state=tk.NORMAL)
        typing_text = f"{self.persona.username} is typing..."
        self.chat_area.insert(tk.END, f"{typing_text}\n", "assistant")
        self.chat_area.config(state=tk.DISABLED)
        self.chat_area.see(tk.END)
        print(f"[DEBUG] Inserted typing indicator.", flush=True)

        # Launch a background thread to get the assistant's response.
        threading.Thread(target=self.get_response, args=(user_message,), daemon=True).start()

    def get_response(self, user_message):
        try:
            print(f"[DEBUG] Generating response for: {user_message}", flush=True)
            prompt, assistant_response = self.conversation.chat(user_message, max_new_tokens=self.max_tokens)
            print(f"prompt: {prompt}")
            print(f"reponse: {assistant_response}")
        except Exception as e:
            assistant_response = "Error generating response."
            print(f"[ERROR] Exception in get_response: {e}", flush=True)
        # Update the typing indicator with the final response on the main thread.
        self.after(0, lambda: self.update_response(assistant_response))

    def update_response(self, response):
        self.chat_area.config(state=tk.NORMAL)
        # Use search to find the typing indicator line.
        typing_index = self.chat_area.search(f"{self.persona.username} is typing...", "1.0", tk.END)
        if typing_index:
            line_number = typing_index.split('.')[0]
            start_index = f"{line_number}.0"
            end_index = f"{line_number}.end"
            # Delete the typing indicator line.
            self.chat_area.delete(start_index, end_index)
            # Insert the actual assistant response at that position.
            self.chat_area.insert(start_index, f"{self.persona.username}: {response}\n", "assistant")
            print(f"[DEBUG] Replaced typing indicator on line {line_number} with response: {response}", flush=True)
        else:
            # If no typing indicator is found, append the response.
            self.chat_area.insert(tk.END, f"{self.persona.username}: {response}\n", "assistant")
            print(f"[DEBUG] Appended response: {response}", flush=True)
        self.chat_area.config(state=tk.DISABLED)
        self.chat_area.see(tk.END)

def main():
    parser = argparse.ArgumentParser(description="Dungeon Master Engine Chat GUI using Tkinter")
    parser.add_argument("--hf_key", type=str, required=True, help="Hugging Face authentication token")
    parser.add_argument("--persona_path", type=str, required=True, help="Path to persona JSON file")
    parser.add_argument("--max_tokens", type=int, default=128, help="Default maximum tokens per generation")
    args = parser.parse_args()

    app = ChatGUI(args.hf_key, args.persona_path, args.max_tokens)
    app.mainloop()

if __name__ == "__main__":
    main()
