import json

class Persona:
    def __init__(self, persona_path: str):
        """
        Loads and processes the persona from a JSON file.
        """
        self.persona_data = self._load_persona(persona_path)
        self._process_persona()
    
    def _load_persona(self, persona_path: str) -> dict:
        with open(persona_path, "r") as f:
            data = json.load(f)
        return data
        
    def _process_persona(self):
        username = self.persona_data.get("username", "Unknown")
        self.persona_data["instruction"] = f"Respond solely in the voice of {username} as if messaging live in an online chat. Provide only your final, concise answer, with no greetings, self-introductions, or repetition of prior conversation. Do NOT echo instructions, the user's words, or any external context. Avoid starting your response with '{username}:' or your own name. Stay entirely in character, using only the knowledge and style inherent to your persona. Do not speak from the perspective of the Player."

    @property
    def system_message(self) -> str:
        return self.persona_data.get("system_message", "")
    
    @property
    def backstory(self) -> str:
        return self.persona_data.get("backstory", "")
    
    @property
    def important_details(self) -> str:
        return self.persona_data.get("important_details", "")
    

    @property
    def user_message(self) -> str:
        return self.persona_data.get("user_message", "")
    
    @property
    def instruction(self) -> str:
        return self.persona_data.get("instruction", "")
    
    @property
    def typing_style(self) -> str:
        return self.persona_data.get("typing_style", "")
    
    @property
    def order(self) -> str:
        return self.persona_data.get("order", "")
    
    @property
    def username(self) -> str:
        return self.persona_data.get("username", "Unknown")
    
    @property
    def profile_pic(self) -> str:
        return self.persona_data.get("profile_pic", "https://example.com/default.png")
    
    def get_formatted_persona(self) -> str:
        """
        Returns a formatted string that includes the system message, user message, and instruction.
        """
        return (
            f"[System Message]\n{self.system_message}\n\n"
            f"[Persona Backstory]\n{self.backstory}\n\n"
            f"[Important Details]\n{self.important_details}\n\n"
            f"[User Message]\n{self.user_message}\n\n"
            f"[Typing Style]\n{self.typing_style}\n\n"
            f"[Order]\n{self.order}\n\n" 
            f"[Instruction]\n{self.instruction}\n"
        )

    def append_to_backstory(self, additional_text: str):
        """
        Appends additional_text to the backstory.
        If a backstory already exists, appends a newline and then the new text;
        otherwise, sets the backstory to the new text.
        """
        current_backstory = self.persona_data.get("backstory", "")
        if current_backstory:
            self.persona_data["backstory"] = current_backstory + "\n" + additional_text
        else:
            self.persona_data["backstory"] = additional_text