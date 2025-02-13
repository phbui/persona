from .validator import Validator
from ...data.record import Record
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
            print(f"[DEBUG] Persona: Loaded persona data from {persona_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load persona from {persona_path}: {e}")
            data = {}
        
        self.setting = data.get("setting", "")
        self.name = data.get("name", "")
        self.backstory = data.get("backstory", "")
        self.goals = data.get("goals", "")
        self.mental_state = data.get("mental_state", {})

        self.training = training
        self.record = Record(self.name)
        self.llm = LLM()
        self.validator = Validator(self, self.llm)
        self.rl = RL(self.name)
        self.emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            tokenizer="j-hartmann/emotion-english-distilroberta-base",
            top_k=None
        )
        self.sentence_transformer = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

        if torch.cuda.is_available():
            self.sentence_transformer.to("cuda")

    def generate_instructions(self):
        return (
            f"You are {self.name}. "
            "You speak solely in the first person and use only your internal knowledge. "
            "You provide only your final, concise answer without greetings, self-introductions, or repetition of prior conversation. "
            "You do NOT echo any instructions, the player's words, external context, or metadata. "
            "You do NOT include any extraneous labels, symbols, or punctuation such as double quotes. "
            "You do NOT output any text enclosed in square brackets. You provide only one response. "
            f"You remain entirely in character as {self.name} and refer only to myself using first-person pronouns. "
            "You do not speak from the perspective of the Player. "
            "You remain true to your character, your knowledge, your notes, and your mental state. "
            "Respond only with what you would say, do not narrate your actions. "
        )

    def update_mental_state(self, changes):
        for key, new_value in changes.items():
            if key in self.mental_state:
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

    def generate_prompt(self, notes, message, history):
        return (
            f"{self.generate_background()}\n\n"
            f"[Your Mental State]\n{self.format_mental_state()}\n\n" 
            f"[Your Current Thoughts]\n{notes}\n\n"
            f"[Recent Conversation]\n{message}\n\n"
            f"[Instructions]\n{self.generate_instructions()}\n\n"
            "[Your Response]\n"
        )
    
    def extract_embeddings(self, message, history):
        context_string = (
            f"{self.generate_background()}\n\n"
            f"[Entire Conversation History]\n{self.format_history(history[:-1])} \n\n"
            f"[Recent Conversation]\n{message}"
        )

        return self.sentence_transformer.encode(context_string)
    
    def extract_emotions(self, message):
        results = self.emotion_classifier(message)
        emotion_scores = results[0]
        emotions = {d['label'].lower(): d['score'] for d in emotion_scores}
             
        return emotions

    def generate_notes(self, message, history):
        prompt_string = (
            f"{self.generate_background()}\n\n"
            f"[Entire Conversation History]\n{self.format_history(history[:-1])}\n\n"
            f"[Current Mental State]\n{self.mental_state}\n\n"
            f"[Recent Conversation]\n{message}\n\n"
            "[Instructions]\n"
            f"Based on the setting, your backstory, your goals, and your current mental state, write a detailed, easy-to-digest summary in the second person as {self.name}. "
            "Your summary should be a bulleted list capturing all names, ideas, topics, and tasks mentioned in the Conversation History. "
            "Do NOT include any new plans, actions, or suggestions—only summarize what is given. "
            "Keep track of you who are talking to, what they said, an where you are. "
            "Do NOT generate any new facts or ideas. \n\n"
            "[Your Notes] \n"
        )

        print("[DEBUG] Persona: Generating notes...")
        return self.llm.generate_response(prompt_string, 512, 0.2)

    def reward_mental_change(self, prev_mental_state, mental_change, history):
        return self.validator.validate_mental_change(prev_mental_state, mental_change, history)

    def reward_notes(self, notes, history):
        return self.validator.validate_notes(notes, history)

    def reward_response(self, response, history):
        return self.validator.validate_response(response, history)

    def reward_response_emotions(self, emotion, history):
        return self.validator.validate_emotions(emotion, history)
    
    def manage_rewards(self, 
                       history, 
                       prev_mental_state, 
                       mental_change, 
                       notes, 
                       response, 
                       response_emotions,
                       offset = 75):
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_mental_change_reward = executor.submit(
                self.reward_mental_change, 
                prev_mental_state, 
                mental_change, 
                history
            )
            future_notes_reward = executor.submit(
                self.reward_notes, 
                notes, 
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

            mental_change_reward = future_mental_change_reward.result() - offset
            notes_reward = future_notes_reward.result() - offset
            response_reward = future_response_reward.result() - offset
            response_emotion_reward = future_response_emotion_reward.result() - offset

        return (mental_change_reward, 
                notes_reward, 
                response_reward, 
                response_emotion_reward)
    
    def _finish_naturally(self, response: str) -> str:
        bracket_index = response.find("[")
        if bracket_index != -1:
            response = response[:bracket_index]

        punctuation_marks = [".", "?", "!"]
        last_index = -1
        for p in punctuation_marks:
            index = response.rfind(p)
            if index > last_index:
                last_index = index
        if last_index != -1:
            return response[:last_index+1].strip()
        return response.strip()
    
    def _count_unique_players(self, history):
       return len({entry["player_name"] for entry in history})


    def generate_response(self, history):
        num_opponents = self._count_unique_players(history)
        messages = history[-num_opponents:]
        message = "\n".join(f"[{msg['player_name']}]: {msg['message']}" for msg in messages)

        embeddings = self.extract_embeddings(message, history)
        emotions = self.extract_emotions(message)

        prev_mental_state = self.mental_state
        mental_change = self.rl.select_action(prev_mental_state, embeddings, emotions)
        self.update_mental_state(mental_change)

        notes = self.generate_notes(message, history)
        prompt = self.generate_prompt(notes, message, history)

        print("[DEBUG] Persona: Generating response...")
        response = self.llm.generate_response(prompt, 128).replace("\n", "").replace("\"", "")
        response = self._finish_naturally(response)

        if self.training: 
            print("[DEBUG] Persona: Updating policy...")
            response_emotions = self.extract_emotions(response)

            mental_change_reward, notes_reward, response_reward, response_emotion_reward = self.manage_rewards(
                history, 
                prev_mental_state, 
                mental_change, 
                notes, 
                response, 
                response_emotions)
            
            self.rl.update_policy(
                mental_change_reward, 
                notes_reward, 
                response_reward, 
                response_emotion_reward)
            
            self.record.record(
                Turn(
                    message,                 
                    embeddings, 
                    emotions, 
                    prev_mental_state,
                    mental_change, 
                    mental_change_reward, 
                    notes,
                    notes_reward,
                    prompt, 
                    response, 
                    response_reward, 
                    response_emotions, 
                    response_emotion_reward,
                    self.rl.policy_net))
            
        return response