import os
import tkinter as tk
from tkinter import scrolledtext, StringVar
from .player import Player  

class PC(Player):
    def __init__(self, name):
        super().__init__(name)
        self.root = None
        self.chat_log = None
        self.message_entry = None
        self.send_button = None
        self.message_var = None  
        self.message_ready = None  # BooleanVar to signal when a message is ready

        # Create the persistent interface immediately on init.
        self.start_chat_interface()

    def start_chat_interface(self):
        if self.root is None:
            # print(f"[DEBUG] Creating chat interface for {self.name}.")
            self.root = tk.Tk()
            self.root.title(f"Chat Interface for {self.name}")

            # Define a custom on_close handler.
            def on_close():
                # print("[DEBUG] Window is closing, exiting...")
                self.root.destroy()
                os._exit(0)
                
            self.root.protocol("WM_DELETE_WINDOW", on_close)

            # Create the scrollable chat log.
            self.chat_log = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, state='disabled', width=80, height=20)
            self.chat_log.pack(padx=10, pady=10)

            # Create a StringVar for the message.
            self.message_var = StringVar()

            # Create a BooleanVar to signal when the message is ready.
            self.message_ready = tk.BooleanVar(value=False)

            # Create the entry widget and link it to the StringVar.
            self.message_entry = tk.Entry(self.root, textvariable=self.message_var, width=80)
            self.message_entry.pack(padx=10, pady=(0, 10))
            # Bind the Return key to _send_message.
            self.message_entry.bind("<Return>", lambda event: self._send_message())

            # Create the Send button.
            self.send_button = tk.Button(self.root, text="Send", command=self._send_message)
            self.send_button.pack(padx=10, pady=(0, 10))

            # Initially disable input.
            self.message_entry.config(state='disabled')
            self.send_button.config(state='disabled')

            # print(f"[DEBUG] Chat interface for {self.name} created.")

    def update_chat_log(self, history):
        # print(f"[DEBUG] {self.name}: Updating chat log.")
        self.chat_log.configure(state='normal')
        self.chat_log.delete("1.0", tk.END)
        for event in history:
            self.chat_log.insert(tk.END, f"{event['player_name']}: {event['message']}\n\n")
        self.chat_log.configure(state='disabled')
        self.chat_log.yview(tk.END)
        # print(f"[DEBUG] {self.name}: Chat log updated.")

    def _send_message(self, event=None):
        if event is not None and event.keysym != "Return":
            return

        current_message = self.message_var.get().strip()
        if current_message:
            # print(f"[DEBUG] {self.name}: _send_message triggered with message: {current_message}")
            # Disable input immediately once a message is sent.
            self.message_entry.config(state='disabled')
            self.send_button.config(state='disabled')
            # Signal that the message is ready so that wait_variable() can continue.
            self.message_ready.set(True)
        # else:
            # print(f"[DEBUG] {self.name}: _send_message triggered but no message was entered.")

    def generate_message(self, history):
        self.update_chat_log(history)
        # print(f"[DEBUG] {self.name}: Enabling input for generate_message.")
        # Enable input.
        self.message_entry.config(state='normal')
        self.send_button.config(state='normal')
        self.message_var.set("")  # Clear any previous message.
        # Reset the flag.
        self.message_ready.set(False)
        # print(f"[DEBUG] {self.name}: Waiting for user input...")
        # Wait here until the message is committed by pressing Enter or Send.
        self.root.wait_variable(self.message_ready)
        message = self.message_var.get().strip()
        # print(f"[DEBUG] {self.name}: generate_message returning: {message}")
        return message
