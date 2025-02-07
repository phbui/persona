from llm import LLM

class Conversation:
    def __init__(self, llm: LLM, persona: str):
        """
        Initializes the conversation with an LLM instance and a persona description.
        The persona description will be prepended to every conversation prompt.
        """
        self.llm = llm
        self.persona = persona
        self.reset_conversation()

    def reset_conversation(self):
        """
        Resets the conversation history while keeping the persona description intact.
        """
        self.history = [f"Persona: {self.persona}"]

    def add_turn(self, speaker: str, text: str):
        """
        Adds a turn to the conversation history.
        Speaker can be "User", "Assistant", or "System".
        """
        self.history.append(f"{speaker}: {text}")

    def get_prompt(self) -> str:
        """
        Combines the conversation history into a single prompt.
        """
        return "\n".join(self.history)

    def chat(self, user_input: str, max_new_tokens: int = 100) -> str:
        """
        Processes a user's input:
          1. Adds the user input to the conversation.
          2. Builds a prompt including the persona and previous turns.
          3. Generates the assistant's response.
          4. Adds the assistant's response to the conversation history.
          5. Returns the response.
        """
        self.add_turn("User", user_input)
        prompt = self.get_prompt()
        response = self.llm.generate_response(prompt, max_new_tokens)
        self.add_turn("Assistant", response)
        return response

    def end_conversation(self) -> str:
        """
        Ends the conversation by adding a termination message.
        Returns the final conversation transcript.
        """
        self.add_turn("System", "The conversation has ended.")
        return self.get_prompt()