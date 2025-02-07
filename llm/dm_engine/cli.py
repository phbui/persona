import argparse
import json
from dm_engine import LLM, Conversation

def load_persona(persona_path: str) -> dict:
    """
    Loads the persona JSON from the given file path and returns it as a dictionary.
    The JSON should include keys like:
      - system_message
      - user_message
      - instruction
      - username
      - profile_pic
    """
    with open(persona_path, "r") as f:
        persona_data = json.load(f)
    return persona_data

def format_persona(persona_data: dict) -> str:
    """
    Returns a formatted persona string to be used as part of the prompt.
    """
    return (
        f"[System Message]\n{persona_data.get('system_message', '')}\n\n"
        f"[User Message]\n{persona_data.get('user_message', '')}\n\n"
        f"[Instruction]\n{persona_data.get('instruction', '')}\n"
    )

def format_response(persona_data: dict, response: str) -> str:
    """
    Prepend the profile picture and username to the response.
    The final response will have the format:
      [(profile_pic)][(username)]: <response>
    """
    username = persona_data.get("username", "Unknown")
    profile_pic = persona_data.get("profile_pic", "https://example.com/default.png")
    return f"[({profile_pic})][({username})]: {response}"

def main():
    parser = argparse.ArgumentParser(description="Dungeon Master Engine CLI")
    
    # Command-line arguments
    parser.add_argument("--hf_key", type=str, required=True, help="Hugging Face authentication token")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["prompt", "conversation"], 
        default="prompt", 
        help="Select mode: 'prompt' for one-off queries, 'conversation' for multi-turn dialogue"
    )
    parser.add_argument(
        "--persona_path", 
        type=str, 
        required=True, 
        help="Path to persona JSON file"
    )
    
    args = parser.parse_args()
    
    # Load the persona data from the JSON file.
    persona_data = load_persona(args.persona_path)
    # Create a formatted persona prompt string.
    persona_prompt = format_persona(persona_data)
    
    # Initialize the LLM interface
    llm_interface = LLM(
        secret_key=args.hf_key, 
        model_name="mistralai/Mistral-7B-Instruct-v0.3", 
    )
    
    # MODE 1: One-off prompt mode
    if args.mode == "prompt":
        print("Prompt mode. Enter your prompt (type 'exit' to quit):")
        while True:
            try:
                prompt = input(">>> ")
                if prompt.lower().strip() == "exit":
                    print("Exiting.")
                    break
                if not prompt.strip():
                    continue  # Skip empty prompts
                
                # Ask for max token limit; default to 100 if input is invalid.
                max_tokens = input("Enter max tokens: ").strip()
                if not max_tokens.isdigit():
                    print("Invalid input. Using default max tokens (100).")
                    max_tokens = 100
                else:
                    max_tokens = int(max_tokens)
                
                print(f"Using max tokens: {max_tokens}")
                
                # Combine the formatted persona prompt with the user's prompt.
                full_prompt = persona_prompt + "\n" + prompt
                
                # Generate the response.
                response = llm_interface.generate_response(full_prompt, max_new_tokens=max_tokens)
                
                print(format_response(persona_data, response))
                print("\n" + "-" * 80 + "\n")
            except KeyboardInterrupt:
                print("\nExiting.")
                break

    # MODE 2: Conversation mode
    elif args.mode == "conversation":
        print("Conversation mode. Press Ctrl+C to end the conversation and see the transcript.")
        
        # Initialize a conversation with the formatted persona prompt.
        conversation = Conversation(llm_interface, persona=persona_prompt)
        
        try:
            while True:
                prompt = input(">>> ")
                if not prompt.strip():
                    continue  # Skip empty prompts
                
                # Ask for max token limit; default to 100 if input is invalid.
                max_tokens = input("Enter max tokens: ").strip()
                if not max_tokens.isdigit():
                    print("Invalid input. Using default max tokens (100).")
                    max_tokens = 100
                else:
                    max_tokens = int(max_tokens)
                
                print(f"Using max tokens: {max_tokens}")
                # Process the user's turn and generate a response.
                response = conversation.chat(prompt, max_new_tokens=max_tokens)
                
                print(format_response(persona_data, response))
                print("\n" + "-" * 80 + "\n")
        except KeyboardInterrupt:
            # When Ctrl+C is pressed, end the conversation gracefully.
            transcript = conversation.end_conversation()
            print("\nConversation ended. Transcript:")
            print(transcript)

if __name__ == "__main__":
    main()
