import argparse
from dm_engine import LLM, Conversation, Persona

def main():
    parser = argparse.ArgumentParser(description="Dungeon Master Engine CLI (Conversation Mode Only)")
    parser.add_argument("--hf_key", type=str, required=True, help="Hugging Face authentication token")
    parser.add_argument("--persona_path", type=str, required=True, help="Path to persona JSON file")
    args = parser.parse_args()


    persona = Persona(args.persona_path)
    
    # Initialize the LLM interface.
    llm_interface = LLM(
        secret_key=args.hf_key,
        model_name="mistralai/Mistral-7B-Instruct-v0.3"
    )
    
    # Initialize the conversation with the persona path (the Conversation class handles all persona processing).
    conversation = Conversation(llm_interface, persona=persona)

    print("Conversation mode. Press Ctrl+C to end the conversation and see the transcript.")
    try:
        while True:
            user_input = input(">>> ")
            if not user_input.strip():
                continue  # Skip empty prompts.
            max_tokens_input = input("Enter max tokens: ").strip()
            if not max_tokens_input.isdigit():
                print("Invalid input. Using default max tokens (100).")
                max_tokens = 100
            else:
                max_tokens = int(max_tokens_input)
            response = conversation.chat(user_input, max_new_tokens=max_tokens)
            
            print(response)
            print("\n" + "-" * 80 + "\n")
    except KeyboardInterrupt:
        transcript = conversation.end_conversation()
        print("\nConversation ended. Transcript:")
        print(transcript)

if __name__ == "__main__":
    main()
