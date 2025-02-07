import numpy as np
from sentence_transformers import SentenceTransformer

class PlayerModel:
    def __init__(self, embedding_dim=384):
        # Initialize conversation history and sentiment counts.
        self.history = []  # List of conversation entries.
        self.sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        self.embedding_dim = embedding_dim
        # Use SentenceTransformer to generate high-quality semantic embeddings.
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def get_embedding(self, text):
        """
        Generate a high-quality embedding from the text using SentenceTransformer.
        """
        return self.encoder.encode(text)
    
    def update(self, message, role):
        """
        Update the model with a new message. This method:
          - Computes the semantic embedding for the message.
          - Determines a simple sentiment label (positive, neutral, negative) based on keyword matching.
          - Updates the sentiment counts.
          - Appends the message, its embedding, and sentiment to the conversation history.
        """
        embedding = self.get_embedding(message)
        sentiment = "neutral"
        text_lower = message.lower()
        if any(word in text_lower for word in ["happy", "good", "great"]):
            sentiment = "positive"
        elif any(word in text_lower for word in ["sad", "bad", "angry"]):
            sentiment = "negative"
        self.sentiment_counts[sentiment] += 1
        self.history.append({
            "role": role,
            "message": message,
            "embedding": embedding,
            "sentiment": sentiment
        })
    
    def get_sentiment_history(self):
        """
        Return a copy of the current sentiment counts.
        """
        return self.sentiment_counts.copy()
    
    def get_behavior_heatmap_data(self):
        """
        Returns a tuple: (list of roles, list of sentiments, heatmap as a numpy array).
        The heatmap counts how many messages of each sentiment have been sent by each role.
        """
        roles = ["player", "llm"]
        sentiments = ["positive", "neutral", "negative"]
        heatmap = np.zeros((len(roles), len(sentiments)))
        for entry in self.history:
            if entry["role"] in roles:
                row = roles.index(entry["role"])
                col = sentiments.index(entry["sentiment"])
                heatmap[row, col] += 1
        return roles, sentiments, heatmap
    
    def get_embedding_projection(self):
        """
        Projects all message embeddings to 2D using PCA.
        Returns a tuple: (embeddings_2d, list of roles corresponding to each embedding).
        If there is no history, returns an empty array and an empty list.
        """
        if not self.history:
            return np.empty((0, 2)), []
        from sklearn.decomposition import PCA
        embeddings = np.array([entry["embedding"] for entry in self.history])
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        roles = [entry["role"] for entry in self.history]
        return embeddings_2d, roles