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
        """
        Injects the username into the instruction if a "{username}" placeholder is found.
        """
        username = self.persona_data.get("username", "Unknown")
        if "instruction" in self.persona_data and "{username}" in self.persona_data["instruction"]:
            self.persona_data["instruction"] = (
                f"Respond solely in the voice of {username}. Do not prefix your response with your username; "
                "simply provide one continuous message that may include ** for actions. Do not include dialogue from any other voices."
            )
    
    @property
    def system_message(self) -> str:
        return self.persona_data.get("system_message", "")
    
    @property
    def user_message(self) -> str:
        return self.persona_data.get("user_message", "")
    
    @property
    def instruction(self) -> str:
        return self.persona_data.get("instruction", "")
    
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
            f"[User Message]\n{self.user_message}\n\n"
            f"[Instruction]\n{self.instruction}\n"
        )
