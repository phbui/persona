import argparse
import tkinter as tk
import threading
from dm_engine import LLM, Conversation, Persona

class ChatGUI(tk.Tk):
    def __init__(self, hf_key, persona_path, max_tokens=128):
        super().__init__()
        self.title("Dungeon Master Engine Chat")
        self.geometry("800x600")
        self.max_tokens = max_tokens

        # Instantiate Persona, LLM, and Conversation.
        self.persona = Persona(persona_path)
        self.llm = LLM(secret_key=hf_key, model_name="mistralai/Mistral-7B-Instruct-v0.3")
        self.conversation = Conversation(self.llm, persona=self.persona)

        # Create a scrollable frame for the chat area.
        self.chat_canvas = tk.Canvas(self, bg="#F0F0F0")
        self.chat_frame = tk.Frame(self.chat_canvas, bg="#F0F0F0")
        self.scrollbar = tk.Scrollbar(self, orient="vertical", command=self.chat_canvas.yview)
        self.chat_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.chat_canvas.pack(side="left", fill="both", expand=True)
        self.chat_canvas.create_window((0, 0), window=self.chat_frame, anchor="nw")
        self.chat_frame.bind("<Configure>", lambda e: self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all")))

        # Create an input field and Send button at the bottom.
        self.entry = tk.Entry(self, font=("Helvetica", 14))
        self.entry.pack(side="bottom", fill="x", padx=10, pady=10)
        self.entry.bind("<Return>", lambda event: self.send_message())
        self.send_button = tk.Button(self, text="Send", font=("Helvetica", 14), command=self.send_message)
        self.send_button.pack(side="bottom")

        # Insert a system join message.
        self.add_message("System", "User joined the chat.", align="center")

    def add_message(self, sender, text, align="left"):
        """
        Adds a message bubble to the chat.
         - For user messages, use align="right".
         - For assistant messages, use align="left".
         - For system messages, use align="center".
        Returns a tuple (frame, message_label).
        """
        bubble_frame = tk.Frame(self.chat_frame, bg="#F0F0F0", pady=5)
        if align == "right":
            bubble_frame.pack(anchor="e", padx=10, pady=5)
        elif align == "left":
            bubble_frame.pack(anchor="w", padx=10, pady=5)
        else:
            bubble_frame.pack(anchor="center", padx=10, pady=5)

        # If not a system message, display the sender's name.
        if sender not in ["System", ""]:
            sender_label = tk.Label(bubble_frame, text=sender, font=("Helvetica", 10, "bold"), bg="#F0F0F0")
            sender_label.pack(anchor="w")
        msg_label = tk.Label(bubble_frame, text=text, font=("Helvetica", 12), bg="#DDEEFF", wraplength=500, justify="left")
        msg_label.pack(anchor="w")
        self.chat_canvas.update_idletasks()
        self.chat_canvas.yview_moveto(1.0)
        return bubble_frame, msg_label

    def send_message(self):
        user_message = self.entry.get().strip()
        if not user_message:
            return
        # Add the user's message (aligned to the right).
        self.add_message("You", user_message, align="right")
        self.entry.delete(0, tk.END)

        # Insert a "typing..." bubble for the assistant (aligned to the left).
        typing_frame, typing_label = self.add_message(self.persona.username, "typing...", align="left")

        # Launch a background thread to get the assistant's response.
        threading.Thread(target=self.get_response, args=(user_message, typing_label), daemon=True).start()

    def get_response(self, user_message, typing_label):
        # Generate the assistant's response using the conversation instance.
        full_response = self.conversation.chat(user_message, max_new_tokens=self.max_tokens)
        # Remove any prefix if present (e.g., "[profile_pic][username]: ").
        if "]: " in full_response:
            assistant_response = full_response.split("]: ", 1)[1]
        else:
            assistant_response = full_response
        # Update the typing bubble with the final response on the main thread.
        self.after(0, lambda: typing_label.config(text=assistant_response))

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