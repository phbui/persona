import json
import numpy as np
from sentence_transformers import SentenceTransformer

class Persona:
    def __init__(self, persona_path: str):
        """
        Loads and processes the persona from a JSON file.
        """
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.persona_data = self._load_persona(persona_path)
        self._process_persona()
        self._process_triggers()

    
    def _load_persona(self, persona_path: str) -> dict:
        with open(persona_path, "r") as f:
            data = json.load(f)
        return data
        
    def _process_persona(self):
        username = self.persona_data.get("username", "Unknown")
        self.persona_data["instruction"] = (
            f"You are {username}, and only {username}. You are messaging a detective who is onto you via an online chat room. "
            "Respond strictly in your own voice—using only your persona's internal knowledge and style. "
            "Provide only your final, concise answer with no greetings, self-introductions, or repetition of prior conversation. "
            "Do NOT echo any instructions, the user's words, or any external context. "
            "Do NOT start your response with '{username}:' or your own name."
            "Remain entirely in character as {username} and do not reference any perspective other than your own. "
            "Do not include any extraneous symbols such as square brackets or meta-commentary. "
            "Respond based on your current mental state. "
        )

        self.persona_data["mental_state"] = {
                "valence": 0,
                "arousal": 0,
                "dominance": 0,
                "confidence": 0,
                "anxiety": 0,
                "guilt": 0,
            }

    def _process_triggers(self):
        self.embedded_triggers = []
        for trigger in self.persona_data.get("triggers", []):
            trig_text = trigger.get("trigger", "")
            changes = trigger.get("changes", {})
            emb = self.encoder.encode(trig_text)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            self.embedded_triggers.append({"embedding": emb, "changes": changes})
                
    def check_triggers(self, embeddings, threshold=1.0):
        for emb in embeddings:
            for trig in self.embedded_triggers.copy():
                distance = np.linalg.norm(emb - trig["embedding"])
                if distance < threshold:
                    print(f"Trigger activated (distance: {distance:.4f}). Applying changes: {trig['changes']}")
                    self.update_mental_state(trig["changes"])

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
    def mental_state(self) -> dict:
        return self.persona_data.get("mental_state", {
                "valence": 0,
                "arousal": 0,
                "dominance": 0,
                "confidence": 0,
                "anxiety": 0,
                "guilt": 0,
            })
    
    @property
    def username(self) -> str:
        return self.persona_data.get("username", "Unknown")
    
    @property
    def profile_pic(self) -> str:
        return self.persona_data.get("profile_pic", "https://example.com/default.png")
    
    @property
    def triggers(self) -> dict:
        return self.persona_data.get("triggers", [])
    
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
            f"[Your Mental State]\n{self.formated_mental_state()}\n\n" 
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

    def formated_mental_state(self):
        """
        Converts a dictionary of mental state parameters into a formatted string.
                                
        Returns:
            str: Formatted string, e.g., "Valence: 0%\nArousal: 0%\n..."
        """
        # Create a list of formatted lines
        formatted_lines = []
        for key, value in self.mental_state.items():
            # Capitalize the key and format the value with a "%" sign
            formatted_lines.append(f"{key.capitalize()}: {value}%")
        
        # Join the lines with a newline character
        return "\n".join(formatted_lines)

    def update_mental_state(self, changes):
        """
        Updates the mental state values based on given changes while ensuring they stay within [0, 10].
        
        :param changes: Dictionary containing the keys of mental states to update and their corresponding delta values.
                        Example: {"guilt": +1, "confidence": -2}
        """
        # Ensure the mental state dictionary exists
        if "mental_state" not in self.persona_data:
            self.persona_data["mental_state"] = {
                "valence": 0,
                "arousal": 0,
                "dominance": 0,
                "confidence": 0,
                "anxiety": 0,
                "guilt": 0,
            }

        for key, delta in changes.items():
            if key in self.persona_data["mental_state"]:
                new_value = self.persona_data["mental_state"][key] + delta
                self.persona_data["mental_state"][key] = max(0, min(100, new_value))

        return self.persona_data["mental_state"]
