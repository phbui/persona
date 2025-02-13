import os
import tkinter as tk
from tkinter import scrolledtext, StringVar
from tkinter import ttk  # Use ttk for modern widget styling
from PIL import Image, ImageTk
from .player import Player  

class PC(Player):
    def __init__(self, name):
        super().__init__(name)
        self.root = None
        self.header_frame = None
        self.chat_log = None
        self.message_entry = None
        self.send_button = None
        self.message_var = None  
        self.message_ready = None  # BooleanVar to signal when a message is ready

        # Create the persistent interface immediately on init.
        self.start_chat_interface()

    def start_chat_interface(self):
        if self.root is None:
            print(f"[DEBUG] Creating chat interface.")
            self.root = tk.Tk()
            self.root.title(f"{self.name}'s Chat Room")
            self.root.geometry("700x600")
            self.root.configure(bg="#ECECEC")
            
            # Optionally, set DPI awareness on Windows.
            try:
                from ctypes import windll
                windll.shcore.SetProcessDpiAwareness(1)
            except Exception as e:
                print("[DEBUG] DPI awareness not set:", e)
                pass

            # Define a custom on_close handler.
            def on_close():
                print("[DEBUG] Window is closing, exiting...")
                self.root.destroy()
                os._exit(0)
            self.root.protocol("WM_DELETE_WINDOW", on_close)

            # Create a header frame with a modern background and title.
            self.header_frame = tk.Frame(self.root, bg="#4CAF50", height=60)
            self.header_frame.pack(fill="x")
            try:
                icon_img = Image.open("images/chat_icon.png")
                icon_img = icon_img.resize((40, 40), Image.ANTIALIAS)
                self.icon_photo = ImageTk.PhotoImage(icon_img)
                icon_label = tk.Label(self.header_frame, image=self.icon_photo, bg="#4CAF50")
                icon_label.pack(side="left", padx=10, pady=10)
            except Exception as e:
                # print("[DEBUG] Icon image not found, skipping icon display.")
                pass
            header_label = tk.Label(self.header_frame, text="Chat", font=("Helvetica", 18, "bold"), fg="white", bg="#4CAF50")
            header_label.pack(side="left", padx=10)
            print(f"[DEBUG] Header for {self.name} created.")

            # Create the scrollable chat log with refined colors and font.
            self.chat_log = scrolledtext.ScrolledText(
                self.root,
                wrap=tk.WORD,
                state='disabled',
                bg="#F5F5F5",
                fg="#333333",
                font=("Helvetica", 12),
                bd=0,
                relief="flat"
            )
            self.chat_log.pack(padx=20, pady=(10,0), fill="both", expand=True)
            print(f"[DEBUG] Chat log for {self.name} created.")

            # Create a frame to hold the message entry and send button.
            input_frame = tk.Frame(self.root, bg="#ECECEC")
            input_frame.pack(padx=20, pady=20, fill="x")

            # Use ttk styling for a modern look.
            style = ttk.Style()
            style.configure("TButton", padding=6, relief="flat", background="#4CAF50",
                            foreground="white", font=("Helvetica", 12, "bold"))
            style.map("TButton", background=[("active", "#45A049")])
            style.configure("TEntry", padding=6, relief="flat", font=("Helvetica", 12))

            # Create a StringVar for the message.
            self.message_var = StringVar()
            # Create a BooleanVar to signal when the message is ready.
            self.message_ready = tk.BooleanVar(value=False)

            # Create the message entry widget using ttk.
            self.message_entry = ttk.Entry(input_frame, textvariable=self.message_var, style="TEntry")
            self.message_entry.grid(row=0, column=0, padx=(0,10), pady=10, sticky="ew")
            input_frame.columnconfigure(0, weight=1)
            # Bind the Return key to _send_message.
            self.message_entry.bind("<Return>", lambda event: self._send_message(event))

            # Create the Send button using ttk.
            self.send_button = ttk.Button(input_frame, text="Send", command=self._send_message)
            self.send_button.grid(row=0, column=1, pady=10)
            print(f"[DEBUG] Input widgets for {self.name} created.")

            # Initially disable input.
            self.message_entry.config(state='disabled')
            self.send_button.config(state='disabled')
            print(f"[DEBUG] Chat interface for {self.name} created.")

    def update_chat_log(self, history):
        print(f"[DEBUG] {self.name}: Updating chat log.")
        self.chat_log.configure(state='normal')
        self.chat_log.delete("1.0", tk.END)
        for event in history:
            # Insert each event with spacing.
            self.chat_log.insert(tk.END, f"{event['player_name']}: {event['message']}\n\n")
        self.chat_log.configure(state='disabled')
        self.chat_log.yview(tk.END)
        print(f"[DEBUG] {self.name}: Chat log updated.")

    def _send_message(self, event=None):
        # If triggered by a key event, only proceed if the Return key was pressed.
        if event is not None and event.keysym != "Return":
            return

        current_message = self.message_var.get().strip()
        if current_message:
            print(f"[DEBUG] {self.name}: _send_message triggered with message: {current_message}")
            # Disable input once a message is sent.
            self.message_entry.config(state='disabled')
            self.send_button.config(state='disabled')
            # Signal that the message is ready.
            self.message_ready.set(True)
        # else:
            print(f"[DEBUG] {self.name}: _send_message triggered but no message was entered.")

    def generate_message(self, history):
        self.update_chat_log(history)
        print(f"[DEBUG] {self.name}: Enabling input for generate_message.")
        # Enable input.
        self.message_entry.config(state='normal')
        self.send_button.config(state='normal')
        self.message_var.set("")  # Clear any previous message.
        # Reset the flag.
        self.message_ready.set(False)
        print(f"[DEBUG] {self.name}: Waiting for user input...")
        # Wait until the user sends a message.
        self.root.wait_variable(self.message_ready)
        message = self.message_var.get().strip()
        print(f"[DEBUG] {self.name}: generate_message returning: {message}")

        return message
      
