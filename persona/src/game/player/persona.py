from .validator import Validator
from ...data.recorder import Recorder
from ...data.turn import Turn
from ...ai.llm import LLM
from ...ai.rl import RL
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
import torch
import json

class Persona():
    def __init__(self, persona_path, training=True):
        try:
            with open(persona_path, "r") as f:
                data = json.load(f)
            print(f"[DEBUG] Loaded persona data from {persona_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load persona from {persona_path}: {e}")
            data = {}
        
        self.setting = data.get("setting", "")
        self.name = data.get("name", "")
        self.backstory = data.get("backstory", "")
        self.goals = data.get("goals", "")
        self.mental_state = data.get("mental_state", {})

        self.training = training
        self.recorder = Recorder(self.name)
        self.llm = LLM()
        self.validator = Validator(self, self.llm)
        self.rl = RL(self.name)
        self.emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            tokenizer="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
        self.sentence_transformer = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

        if torch.cuda.is_available():
            self.sentence_transformer.to("cuda")

    def generate_instructions(self):
        return (
            f"You are {self.name}, and only {self.name}. "
            "Respond strictly in your own voice—using only your persona's internal knowledge. "
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
    
    def generate_background(self):
        return (            
            f"[Setting]\n{self.setting}\n\n"
            f"[Your Name]\n{self.name}\n\n"
            f"[Your Backstory]\n{self.backstory}\n\n"
            f"[Your Goals]\n{self.goals}\n\n"
            )

    def generate_prompt(self, focus):
        return (
            f"{self.generate_background()}\n\n"
            f"[Your Mental State]\n{self.format_mental_state()}\n\n" 
            f"[Your Focus]\n{focus}\n\n"
            f"[Instructions]\n{self.generate_instructions}\n\n"
            "[How do you answer?]\n"
        )
    
    def extract_embeddings(self, message, history):
        context_string = (
            f"{self.generate_background()}\n\n"
            f"[Your Conversation So Far]\n{self.format_history(history[:-1])} \n\n"
            f"[Player Message]\n{message}"
        )

        return self.sentence_transformer.encode(context_string)
    
    def extract_emotions(self, message):
        results = self.emotion_classifier(message)
        emotion_scores = results[0]
        emotions = {d['label'].lower(): d['score'] for d in emotion_scores}
             
        return emotions

    def generate_focus(self, history):
        prompt_string = (
            f"{self.generate_background()}\n\n"
            f"[Your Conversation So Far]\n{self.format_history(history)} \n\n"
            f"[Your Mental State]\n{self.mental_state}"
            f"[Instructions] Based on the above information, write in the second person as {self.name} describing what you are thinking right now. "
            f"Keep it concise, reflective, and true to your character and mental state."
        )

        return self.llm.generate_response(prompt_string, 128)

    def reward_mental_change(self, prev_mental_state, mental_change, history):
        return self.validator.validate_mental_change(prev_mental_state, mental_change, history)

    def reward_focus(self, focus, history):
        return self.validator.validate_focus(focus, history)

    def reward_response(self, response, history):
        return self.validator.validate_response(response, history)

    def reward_response_emotions(self, emotion, history):
        return self.validator.validate_emotion(emotion, history)
    
    def manage_rewards(self, 
                       history, 
                       prev_mental_state, 
                       mental_change, 
                       focus, 
                       response, 
                       response_emotions):
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_mental_change_reward = executor.submit(
                self.reward_mental_change, 
                prev_mental_state, 
                mental_change, 
                history
            )
            future_focus_reward = executor.submit(
                self.reward_focus, 
                focus, 
                history
            )
            future_response_reward = executor.submit(
                self.reward_response, 
                response, 
                history
            )
            future_response_emotion_reward = executor.submit(
                self.reward_response_emotions, 
                response_emotions, 
                history
            )

            mental_change_reward = future_mental_change_reward.result()
            focus_reward = future_focus_reward.result()
            response_reward = future_response_reward.result()
            response_emotion_reward = future_response_emotion_reward.result()

        return (mental_change_reward, 
                focus_reward, 
                response_reward, 
                response_emotion_reward)

    def generate_response(self, history):
        message = history[-1]['message']

        embeddings = self.extract_embeddings(message, history)
        emotions = self.extract_emotions(message)

        prev_mental_state = self.mental_state
        mental_change = self.rl.select_action(prev_mental_state, embeddings, emotions)
        self.update_mental_state(mental_change)

        focus = self.generate_focus(history)
        prompt = self.generate_prompt(focus)

        response = self.llm.generate_response(prompt)

        if self.training: 
            response_emotions = self.extract_emotions(response)

            mental_change_reward, focus_reward, response_reward, response_emotion_reward = self.manage_rewards(
                history, 
                prev_mental_state, 
                mental_change, 
                focus, 
                response, 
                response_emotions)
            
            self.rl.update_policy(
                mental_change_reward, 
                focus_reward, 
                response_reward, 
                response_emotion_reward)
        
            self.recorder.record(
                Turn(
                    message,                 
                    embeddings, 
                    emotions, 
                    mental_change, 
                    mental_change_reward, 
                    focus,
                    focus_reward,
                    prompt, 
                    response, 
                    response_reward, 
                    response_emotions, 
                    response_emotion_reward,
                    self.rl.policy_net))
            
        return response