import argparse
import pygame
from dm_engine import LLM, Conversation, Persona

def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(
        description="Dungeon Master Engine GUI using Pygame (Conversation Mode)"
    )
    parser.add_argument("--hf_key", type=str, required=True, help="Hugging Face authentication token")
    parser.add_argument("--persona_path", type=str, required=True, help="Path to persona JSON file")
    parser.add_argument("--max_tokens", type=int, default=128, help="Default maximum tokens per generation")
    args = parser.parse_args()

    # Instantiate Persona, LLM, and Conversation.
    persona = Persona(args.persona_path)
    llm_interface = LLM(secret_key=args.hf_key, model_name="mistralai/Mistral-7B-Instruct-v0.3")
    conversation = Conversation(llm_interface, persona=persona)

    # Initialize Pygame.
    pygame.init()
    screen_width, screen_height = 800, 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Dungeon Master Engine GUI")
    clock = pygame.time.Clock()

    # Set up fonts and colors.
    font = pygame.font.SysFont("Courier New", 20)
    input_font = pygame.font.SysFont("Courier New", 24)
    text_color = (255, 255, 255)  # white
    bg_color = (0, 0, 0)          # black

    # The transcript stores lines of conversation.
    transcript = []
    transcript.append("Welcome to Dungeon Master Engine!")
    transcript.append("Type your message and press Enter.")

    input_text = ""  # Current input field content.

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Handle keyboard events for text input.
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                elif event.key == pygame.K_RETURN:
                    # When Enter is pressed, process the input.
                    if input_text.strip():
                        transcript.append(">>> " + input_text)
                        # Generate response from the conversation.
                        response = conversation.chat(input_text, max_new_tokens=args.max_tokens)
                        transcript.append(response)
                        input_text = ""  # Clear input after sending.
                else:
                    input_text += event.unicode

        # Draw background.
        screen.fill(bg_color)

        # Render the transcript.
        y_offset = 10
        line_height = font.get_height() + 5
        # Calculate how many lines can fit in the transcript area (leaving space for the input field).
        max_lines = (screen_height - 50) // line_height
        for line in transcript[-max_lines:]:
            rendered_line = font.render(line, True, text_color)
            screen.blit(rendered_line, (10, y_offset))
            y_offset += line_height

        # Render the input field.
        input_surface = input_font.render(">> " + input_text, True, text_color)
        screen.blit(input_surface, (10, screen_height - 40))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()
