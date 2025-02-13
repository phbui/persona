class Turn:
    def __init__(self, 
                 input_message, 
                 input_message_embeddings, 
                 input_message_emotions, 
                 prev_mental_state,
                 mental_change, 
                 reward_mental_change, 
                 notes,
                 notes_reward,
                 prompt, 
                 response, 
                 response_reward, 
                 response_emotion,
                 response_emotion_reward,
                 policy):
        self.input_message = input_message
        self.input_message_embedding = input_message_embeddings
        self.input_message_emotion = input_message_emotions
        self.prev_mental_state = prev_mental_state
        self.mental_change = mental_change
        self.reward_mental_change = reward_mental_change
        self.notes = notes
        self.notes_reward = notes_reward
        self.prompt = prompt
        self.response = response
        self.response_reward = response_reward
        self.response_emotion = response_emotion
        self.response_emotion_reward = response_emotion_reward
        self.policy = policy

    def __str__(self):
        return (
            f"Turn(\n"
            f"  input_message: {self.input_message}\n"
            f"  input_message_embedding: {self.input_message_embedding}\n"
            f"  input_message_emotion: {self.input_message_emotion}\n"
            f"  prev_mental_state: {self.prev_mental_state}\n"
            f"  mental_change: {self.mental_change}\n"
            f"  reward_mental_change: {self.reward_mental_change}\n"
            f"  notes: {self.notes}\n"
            f"  notes_reward: {self.notes_reward}\n"
            f"  prompt: {self.prompt}\n"
            f"  response: {self.response}\n"
            f"  response_reward: {self.response_reward}\n"
            f"  response_emotion: {self.response_emotion}\n"
            f"  response_emotion_reward: {self.response_emotion_reward}\n"
            f"  policy: {self.policy}\n"
            f")"
        )