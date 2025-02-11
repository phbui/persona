class Turn:
    def __init__(self, 
                 input_message, 
                 input_message_embeddings, 
                 input_message_emotions, 
                 mental_change, 
                 reward_mental_change, 
                 focus,
                 focus_reward,
                 prompt, 
                 response, 
                 response_reward, 
                 response_emotion,
                 response_emotion_reward,
                 policy):
        self.input_message = input_message
        self.input_message_embedding = input_message_embeddings
        self.input_message_emotion = input_message_emotions
        self.mental_change = mental_change
        self.reward_mental_change = reward_mental_change
        self.focus = focus
        self.focus_reward = focus_reward
        self.prompt = prompt
        self.response = response
        self.response_reward = response_reward
        self.response_emotion = response_emotion
        self.response_emotion_reward = response_emotion_reward,
        self.policy = policy