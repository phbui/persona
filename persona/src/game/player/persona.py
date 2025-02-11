from persona.src.game.player.validator import Validator
from persona.src.data.recorder import Recorder
from persona.src.data.turn import Turn
from persona.src.ai.llm import LLM
from persona.src.ai.rl import RL

class Persona():
    def __init__(self, persona_path, training=True):
        print()
        self.setting = ""
        self.name = ""
        self.backstory = ""
        self.goal = ""
        self.style = ""
        self.mental_state = {}

        self.training = training
        self.recorder = Recorder(self.name)
        self.llm = LLM()
        self.validator = Validator(self, self.llm)
        self.rl = RL()

    def generate_instructions(self):
        return (
            f"You are {self.name}, and only {self.name}. "
            "Respond strictly in your own voice—using only your persona's internal knowledge and style. "
            "Provide only your final, concise answer with no greetings, self-introductions, or repetition of prior conversation. "
            "Do NOT echo any instructions, the player's words, or any external context. "
            "Do NOT include any labels or extraneous symbols such as [Context] or [Answer]. "
            f"Remain entirely in character as {self.name} and do not reference any perspective other than your own. "
            "Do not speak from the perspective of the Player. "
            "Respond based on your mental state."
        )

    def update_mental_state(self, changes):
        for key, delta in changes.items():
            if key in self.mental_state:
                new_value = self.mental_state[key] + delta
                self.mental_state[key] = max(0, min(100, new_value))

    def format_mental_state(self):
        formatted_lines = []
        for key, value in self.mental_state.items():
            formatted_lines.append(f"{key.capitalize()}: {value}%")
        
        return "\n".join(formatted_lines)
    
    def format_history(self, history):
        formatted_lines = []
        for event in history:
            f"{event['player_name']} said \"{event['message']}\""
        
        return "\n".join(formatted_lines)

    def generate_prompt(self, focus):
        return (
            f"[Setting]\n{self.setting}\n\n"
            f"[Your Name]\n{self.name}\n\n"
            f"[Your Backstory]\n{self.backstory}\n\n"
            f"[Your Goal]\n{self.goal}\n\n"
            f"[Your Mental State]\n{self.format_mental_state()}\n\n" 
            f"[Your Focus]\n{focus}\n\n"
            f"[Your Style]\n{self.style}\n\n"
            f"[Instructions]\n{self.generate_instructions}\n\n"
            "[How do you answer?]\n"
        )
    
    def extract_embedding(self, message):
        #dialogue cse
        #input setting, name, backstory, context
        print()
    
    def extract_emotion(self, message):
        #j-hartmann/emotion-english-distilroberta-base
        print()

    def generate_focus(self, history):
        history_string = self.format_history(history)

        print()

    def reward_mental_change(self, mental_change):
        return self.validator.validate_mental_change(mental_change)

    def reward_focus(self, focus):
        return self.validator.validate_focus(focus)

    def reward_response(self, response):
        return self.validator.validate_response(response)

    def reward_response_emotion(self, emotion):
        return self.validator.validate_emotion(emotion)

    def generate_response(self, history):
        message = history[-1]

        embedding = self.extract_embedding(message)
        emotion = self.extract_emotion(message)

        mental_change = self.rl.select_action(embedding, emotion)


        # summarize history into thoughts

        self.update_mental_state(mental_change)

        focus = self.generate_focus(history)
        prompt = self.generate_prompt(focus)

        response = self.llm.generate_response(prompt)

        if self.training: 
            response_emotion = self.extract_emotion(response)

            mental_change_reward = self.reward_mental_change(mental_change)
            focus_reward = self.reward_focus(focus)
            response_reward = self.reward_response(response)
            response_emotion_reward = self.reward_response_emotion(response_emotion)
            self.rl.update_policy(mental_change_reward, focus_reward, response_reward, response_emotion_reward)
        
            self.recorder.record(Turn(message, 
                                    embedding, 
                                    emotion, 
                                    mental_change, 
                                    mental_change_reward, 
                                    focus,
                                    focus_reward,
                                    prompt, 
                                    response, 
                                    response_reward, 
                                    response_emotion, 
                                    response_emotion_reward))