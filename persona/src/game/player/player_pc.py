import tkinter as tk
from tkinter import scrolledtext
from tkinter import StringVar
from .player import Player  

class PC(Player):
    def __init__(self, name):
        super().__init__(name)
        self.root = None
        self.chat_log = None
        self.message_entry = None
        self.send_button = None
        self.message_var = None  

        # Create the persistent interface immediately on init.
        self.start_chat_interface()

    def start_chat_interface(self):
        if self.root is None:
            print(f"[DEBUG] Creating chat interface for {self.name}.")
            self.root = tk.Tk()
            self.root.title(f"Chat Interface for {self.name}")

            # Create the scrollable chat log.
            self.chat_log = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, state='disabled', width=80, height=20)
            self.chat_log.pack(padx=10, pady=10)

            # Create a StringVar for the message.
            self.message_var = StringVar()

            # Create the entry widget and link it to the StringVar.
            self.message_entry = tk.Entry(self.root, textvariable=self.message_var, width=80)
            self.message_entry.pack(padx=10, pady=(0, 10))
            # Bind the Return key to _send_message.
            self.message_entry.bind("<Return>", lambda event: self._send_message())

            # Create the Send button.
            self.send_button = tk.Button(self.root, text="Send", command=lambda: self._send_message())
            self.send_button.pack(padx=10, pady=(0, 10))

            # Initially disable input.
            self.message_entry.config(state='disabled')
            self.send_button.config(state='disabled')

            print(f"[DEBUG] Chat interface for {self.name} created.")

    def update_chat_log(self, history):
        """
        Updates the chat log widget with the current conversation history.
        Expects 'history' to be a list of dictionaries with keys "player_name" and "message".
        """
        print(f"[DEBUG] {self.name}: Updating chat log.")
        self.chat_log.configure(state='normal')
        self.chat_log.delete("1.0", tk.END)
        for event in history:
            # Format each event; for example: "Alice said "Hello!""
            self.chat_log.insert(tk.END, f"{event['player_name']} said \"{event['message']}\"\n")
        self.chat_log.configure(state='disabled')
        self.chat_log.yview(tk.END)
        print(f"[DEBUG] {self.name}: Chat log updated.")

    def _send_message(self):
        """
        Internal method triggered by pressing Enter or clicking Send.
        It does not return anything but signals that input has been provided.
        Once a message is entered, it disables the input to prevent further typing until next generate_message call.
        """
        current_message = self.message_var.get().strip()
        if current_message:
            print(f"[DEBUG] {self.name}: _send_message triggered with message: {current_message}")
            # Disable input immediately once a message is sent.
            self.message_entry.config(state='disabled')
            self.send_button.config(state='disabled')
        else:
            print(f"[DEBUG] {self.name}: _send_message triggered but no message was entered.")

    def generate_message(self, history):
        """
        Called by the game to get a message from the user.
        1. Updates the chat log with the current history.
        2. Enables input so the user can type.
        3. Waits until the user enters and submits a message.
        4. Returns the message and then disables input.
        """
        # Update chat log to reflect the latest history.
        self.update_chat_log(history)
        print(f"[DEBUG] {self.name}: Enabling input for generate_message.")
        
        # Enable the message entry and send button.
        self.message_entry.config(state='normal')
        self.send_button.config(state='normal')
        
        # Clear any previous input.
        self.message_var.set("")
        
        print(f"[DEBUG] {self.name}: Waiting for user input...")
        # Block until the message_var is updated by _send_message.
        self.root.wait_variable(self.message_var)
        
        # When wait_variable returns, retrieve the message.
        message = self.message_var.get().strip()
        print(f"[DEBUG] {self.name}: generate_message returning: {message}")
        return message

