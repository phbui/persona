import re

class Validator:
    def __init__(self, persona, llm):
        self.persona = persona
        self.llm = llm

    def extract_numeric_score(self, response: str) -> int:
        cleaned = response.strip()
        match = re.search(r'\d+', cleaned)
        if not match:
            raise ValueError("No numeric value found")
        num = int(match.group())
        return max(0, min(num, 100))
    
    def _validate(self,
                  history, 
                  sections: dict, 
                  instruction: str) -> int:
        prompt_parts = []
        prompt_parts.append(self.persona.generate_background())
        prompt_parts.append(f"[Conversation So Far]\n{self.persona.format_history(history)}\n\n")
        for header, content in sections.items():
            prompt_parts.append(f"[{header}]\n{content}\n\n")

        base_instructions = " Be harsh but fair with grading. Be very precise down to the hundredth's place. Your response must include only the integer, with no additional text, whitespace, or punctuation.\n\n[Answer]\n"
        instruction = instruction + base_instructions
        prompt_parts.append(f"[Instructions]\n{instruction}")
        prompt_string = "\n\n".join(prompt_parts)
        score_response = self.llm.generate_response(prompt_string, 6)
        return self.extract_numeric_score(score_response)
    
    def validate_mental_change(self, prev_mental_state, mental_change, history):
        sections = {
            "Your PREVIOUS Mental State": self.format_mental_state(prev_mental_state),
            "Your NEW Mental State": self.format_mental_state(mental_change)
        }
        instruction = ("Based on all of the above information, respond with only a single integer "
                       "between 0 and 100 that represents how accurate your new mental state is for your character.")
        
        print("[DEBUG] Validator: Validating mental state...")
        return self._validate(history, sections, instruction)
    
    def validate_notes(self, notes, history):
        sections = {
            "Your notes": notes
        }
        instruction = ("Based on all of the above information, respond with only a single integer "
                       "between 0 and 100 that represents how accurate your notes are for your character.")
        
        print("[DEBUG] Validator: Validating notes...")
        return self._validate(history, sections, instruction)
    
    def validate_response(self, response, history):
        sections = {
            "Your Response": response
        }
        instruction = ("Based on all of the above information, respond with only a single integer "
                       "between 0 and 100 that represents how accurate your response is for your character.")
        
        print("[DEBUG] Validator: Validating response...")
        return self._validate(history, sections, instruction)
    
    def validate_emotions(self, emotions, history):
        sections = {
            "Your Emotions": self.format_emotions(emotions)
        }
        instruction = ("Based on all of the above information, respond with only a single integer "
                       "between 0 and 100 that represents how accurate your emotions are for your character.")
        
        print("[DEBUG] Validator: Validating emotions...")
        return self._validate(history, sections, instruction)
    
    def format_emotions(self, emotions: dict) -> str:
        output_lines = []
        for emotion, value in emotions.items():
            if value >= 0.9:
                intensity = "overwhelmingly"
            elif value >= 0.8:
                intensity = "extremely"
            elif value >= 0.7:
                intensity = "very"
            elif value >= 0.6:
                intensity = "quite"
            elif value >= 0.5:
                intensity = "moderately"
            elif value >= 0.4:
                intensity = "somewhat"
            elif value >= 0.3:
                intensity = "mildly"
            elif value >= 0.2:
                intensity = "slightly"
            elif value >= 0.1:
                intensity = "barely"
            else:
                intensity = "hardly"
            output_lines.append(f"I feel {intensity} {emotion}.")
        return " ".join(output_lines)

    def format_mental_state(self, mental_state: dict) -> str:
        output_lines = []
        for state, value in mental_state.items():
            if value >= 90:
                descriptor = "exceptional"
            elif value >= 80:
                descriptor = "excellent"
            elif value >= 70:
                descriptor = "very good"
            elif value >= 60:
                descriptor = "good"
            elif value >= 50:
                descriptor = "fair"
            elif value >= 40:
                descriptor = "mediocre"
            elif value >= 30:
                descriptor = "poor"
            elif value >= 20:
                descriptor = "very poor"
            elif value >= 10:
                descriptor = "extremely poor"
            else:
                descriptor = "abysmal"
            output_lines.append(f"My mental state for {state} is {descriptor} ({value}).")
        return " ".join(output_lines)
