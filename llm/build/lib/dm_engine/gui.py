import argparse
from chat import Chat

def main():
    parser = argparse.ArgumentParser(
        description="Dungeon Master Engine Chat GUI with Integrated Player Model Visualization")
    parser.add_argument("--hf_key", type=str, required=True, help="Hugging Face authentication token")
    parser.add_argument("--persona_path", type=str, required=True, help="Path to persona JSON file")
    parser.add_argument("--max_tokens", type=int, default=128, help="Default maximum tokens per generation")
    args = parser.parse_args()

    print("[DEBUG] Starting chat.")
    app = Chat(args.hf_key, args.persona_path, args.max_tokens)
    app.mainloop()

    conversation_data = app.conversation_end_data
    print("Retrieved conversation data:", conversation_data)

    # Now you can manipulate 'conversation_data' as needed.
    # For example:
    if conversation_data is not None:
        # Process the data (e.g., save to a file, analyze, etc.)
        with open("conversation_end_data.txt", "w") as f:
            f.write(str(conversation_data))

if __name__ == "__main__":
    main()
