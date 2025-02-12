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

        base_instructions = " Your response must include only the integer, with no additional text, whitespace, or punctuation.\n\n[Answer]\n"
        instruction = instruction + base_instructions
        prompt_parts.append(f"[Instructions]\n{instruction}")
        prompt_string = "\n\n".join(prompt_parts)
        score_response = self.llm.generate_response(prompt_string, 6)
        # print(f"Score: {score_response}")
        return self.extract_numeric_score(score_response)
    
    def validate_mental_change(self, prev_mental_state, mental_change, history):
        sections = {
            "Your PREVIOUS Mental State": prev_mental_state,
            "Your NEW Mental State": mental_change
        }
        instruction = ("Based on all of the above information, respond with only a single integer "
                       "between 0 and 100 that represents how accurate your new mental state is for your character.")
        
        # print("Validating mental state...")
        return self._validate(history, sections, instruction)
    
    def validate_focus(self, focus, history):
        sections = {
            "Your Focus": focus
        }
        instruction = ("Based on all of the above information, respond with only a single integer "
                       "between 0 and 100 that represents how accurate your focus is for your character.")
        
        # print("Validating focus...")
        return self._validate(history, sections, instruction)
    
    def validate_response(self, response, history):
        sections = {
            "Your Response": response
        }
        instruction = ("Based on all of the above information, respond with only a single integer "
                       "between 0 and 100 that represents how accurate your response is for your character.")
        
        # print("Validating response...")
        return self._validate(history, sections, instruction)
    
    def validate_emotions(self, emotions, history):
        sections = {
            "Your Emotions": emotions
        }
        instruction = ("Based on all of the above information, respond with only a single integer "
                       "between 0 and 100 that represents how accurate your emotions are for your character.")
        
        # print("Validating emotions...")
        return self._validate(history, sections, instruction)
