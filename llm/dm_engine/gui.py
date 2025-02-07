import argparse
import tkinter as tk
from tkinter import scrolledtext
from dm_engine import LLM, Conversation, Persona

def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(
        description="Dungeon Master Engine Chat GUI using Tkinter (Conversation Mode)"
    )
    parser.add_argument("--hf_key", type=str, required=True, help="Hugging Face authentication token")
    parser.add_argument("--persona_path", type=str, required=True, help="Path to persona JSON file")
    parser.add_argument("--max_tokens", type=int, default=128, help="Default maximum tokens per generation")
    args = parser.parse_args()

    # Instantiate Persona, LLM, and Conversation.
    persona = Persona(args.persona_path)
    llm_interface = LLM(secret_key=args.hf_key, model_name="mistralai/Mistral-7B-Instruct-v0.3")
    conversation = Conversation(llm_interface, persona=persona)

    # Create the main Tkinter window.
    root = tk.Tk()
    root.title("Dungeon Master Engine Chat")
    root.geometry("800x600")

    # Create a scrolled text widget for the chat history.
    chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, state="disabled", font=("Helvetica", 12))
    chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    # Create a frame for the input field and Send button.
    input_frame = tk.Frame(root)
    input_frame.pack(padx=10, pady=10, fill=tk.X)
    input_field = tk.Entry(input_frame, font=("Helvetica", 12))
    input_field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
    send_button = tk.Button(input_frame, text="Send", font=("Helvetica", 12), command=lambda: send_message())
    send_button.pack(side=tk.LEFT)

    def insert_message(sender, message):
        """Insert a message into the chat area."""
        chat_area.config(state="normal")
        chat_area.insert(tk.END, f"{sender}: {message}\n")
        chat_area.config(state="disabled")
        chat_area.see(tk.END)

    # Display a join message when the chat starts.
    insert_message("System", "User joined the chat.")

    def send_message():
        user_message = input_field.get().strip()
        if not user_message:
            return
        # Insert the user's message.
        insert_message("You", user_message)
        input_field.delete(0, tk.END)
        
        # Insert a typing indicator for the assistant.
        chat_area.config(state="normal")
        # Record the current number of lines (the last line index)
        lines_before = int(chat_area.index('end-1c').split('.')[0])
        chat_area.insert(tk.END, f"{persona.username} is typing...\n")
        chat_area.config(state="disabled")
        chat_area.see(tk.END)
        
        # Generate the assistant's response (blocking call; consider threading if needed)
        full_response = conversation.chat(user_message, max_new_tokens=args.max_tokens)
        # conversation.chat returns a formatted string like "[profile_pic][username]: response"
        if "]: " in full_response:
            assistant_response = full_response.split("]: ", 1)[1]
        else:
            assistant_response = full_response
        
        # Remove the "typing..." indicator.
        chat_area.config(state="normal")
        # Delete the line that was just inserted.
        start_index = f"{lines_before}.0"
        end_index = f"{lines_before + 1}.0"
        chat_area.delete(start_index, end_index)
        chat_area.config(state="disabled")
        
        # Insert the assistant's actual response.
        insert_message(persona.username, assistant_response)

    # Bind the Return key to send the message.
    root.bind("<Return>", lambda event: send_message())

    # Start the Tkinter event loop.
    root.mainloop()

if __name__ == "__main__":
    main()
