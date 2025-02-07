import argparse
from chat import Chat

def main():
    parser = argparse.ArgumentParser(
        description="Dungeon Master Engine Chat GUI with Integrated Player Model Visualization")
    parser.add_argument("--hf_key", type=str, required=True, help="Hugging Face authentication token")
    parser.add_argument("--persona_path", type=str, required=True, help="Path to persona JSON file")
    parser.add_argument("--max_tokens", type=int, default=128, help="Default maximum tokens per generation")
    args = parser.parse_args()

    print("[DEBUG] Starting ChatGUI application.")
    app = Chat(args.hf_key, args.persona_path, args.max_tokens)
    app.mainloop()

if __name__ == "__main__":
    main()
