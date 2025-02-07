from llm import LLM
from persona import Persona

class Conversation:
    def __init__(self, llm: LLM, persona: Persona):
        """
        Initializes the conversation with an LLM instance and a Persona instance.
        The Conversation class uses the Persona object for all context and formatting.
        """
        self.llm = llm
        self.persona_obj = persona
        self.persona = self.persona_obj.get_formatted_persona()
        self.username = self.persona_obj.username
        self.profile_pic = self.persona_obj.profile_pic
        self.history = []  # Will store conversation turns with speaker labels (for context only)
    
    def reset_conversation(self):
        """Resets the conversation history."""
        self.history = []
    
    def add_turn(self, speaker: str, text: str):
        """
        Adds a turn to the conversation history with speaker labels for internal context.
        (This data is used to build the full prompt but is hidden from the final answer.)
        """
        self.history.append(f"{speaker}: {text.strip()}")
    
    def get_prompt(self) -> str:
        """
        Combines the formatted persona and conversation history into a single prompt.
        The history is wrapped within <CONTEXT> markers so that it serves as background context
        without being repeated verbatim in the final answer.
        """
        context = "\n".join(self.history)
        prompt = (
            f"{self.persona}\n\n"
            f"<CONTEXT>\n{context}\n</CONTEXT>\n\n"
            "Answer:"
        )
        return prompt
    
    def _finish_naturally(self, response: str) -> str:
        """
        Attempts to ensure the response ends naturally.
        It looks for the last occurrence of common sentence-ending punctuation (., ?, !)
        and trims the response there.
        """
        punctuation_marks = [".", "?", "!"]
        last_index = -1
        for p in punctuation_marks:
            index = response.rfind(p)
            if index > last_index:
                last_index = index
        if last_index != -1:
            return response[:last_index+1].strip()
        return response.strip()
    
    def chat(self, user_input: str, max_new_tokens: int = 100) -> str:
        """
        Processes a user's input:
          1. Adds the user's input (labeled as 'Player') to the conversation history.
          2. Builds a prompt including the formatted persona and the hidden conversation context.
          3. Generates the assistant's response.
          4. Trims the response to finish naturally.
          5. Adds the assistant's response (labeled with the persona's username) to the conversation history.
          6. Returns the final, formatted response prefixed with the profile picture and username.
        """
        self.add_turn("Player", user_input)
        prompt = self.get_prompt()
        response = self.llm.generate_response(prompt, max_new_tokens)
        response = self._finish_naturally(response)
        self.add_turn(self.username, response)
        final_response = f"[{self.profile_pic}][{self.username}]: {response}"
        return final_response
    
    def end_conversation(self) -> str:
        """
        Ends the conversation by appending a termination message.
        Returns the full conversation transcript (including the hidden context).
        """
        self.add_turn("System", "The conversation has ended.")
        return self.get_prompt()
