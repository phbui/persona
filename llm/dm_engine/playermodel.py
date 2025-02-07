import numpy as np
from sklearn.decomposition import PCA
import networkx as nx

class PlayerModel:
    def __init__(self, embedding_dim=50):
        # List to hold conversation history entries
        self.history = []  # Each entry: dict with keys "role", "message", "embedding", "sentiment"
        # Sentiment counts for a probabilistic view (simplified: positive, neutral, negative)
        self.sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        self.embedding_dim = embedding_dim
        # Last predicted embedding (for reward computation)
        self.last_prediction = None

    def get_embedding(self, text):
        """
        Dummy embedding function: here we use a deterministic hash-based method
        to simulate an embedding vector of size embedding_dim.
        """
        rng = np.random.RandomState(hash(text) % 1234567)
        return rng.rand(self.embedding_dim)

    def analyze_sentiment(self, text):
        """
        Dummy sentiment analysis: very simple keyword-based.
        """
        text_lower = text.lower()
        if any(word in text_lower for word in ["happy", "great", "good", "excited"]):
            return "positive"
        elif any(word in text_lower for word in ["sad", "bad", "angry", "upset"]):
            return "negative"
        else:
            return "neutral"

    def update(self, message, role):
        """Update the model with a new message from either the user or the assistant."""
        embedding = self.get_embedding(message)
        sentiment = self.analyze_sentiment(message)
        # Update sentiment counts
        if sentiment in self.sentiment_counts:
            self.sentiment_counts[sentiment] += 1
        else:
            self.sentiment_counts[sentiment] = 1

        # Save the conversation entry
        entry = {
            "role": role,
            "message": message,
            "embedding": embedding,
            "sentiment": sentiment
        }
        self.history.append(entry)

    def predict_next_response(self):
        """
        Predict the next response by taking an average of all embeddings
        and adding a small noise term. This is a stand-in for a more sophisticated
        prediction mechanism using the LLM and embedding model.
        """
        if not self.history:
            prediction = np.zeros(self.embedding_dim)
        else:
            embeddings = np.array([entry["embedding"] for entry in self.history])
            prediction = embeddings.mean(axis=0)
        # Add small random noise to simulate uncertainty.
        prediction += np.random.normal(0, 0.01, size=self.embedding_dim)
        self.last_prediction = prediction
        return prediction

    def compute_reward(self, actual_response):
        """
        Compute a reward based on the cosine similarity between the predicted
        embedding and the actual response embedding.
        """
        if self.last_prediction is None:
            return 0.0

        actual_embedding = self.get_embedding(actual_response)
        # Cosine similarity
        dot = np.dot(self.last_prediction, actual_embedding)
        norm_pred = np.linalg.norm(self.last_prediction)
        norm_actual = np.linalg.norm(actual_embedding)
        if norm_pred == 0 or norm_actual == 0:
            reward = 0.0
        else:
            reward = dot / (norm_pred * norm_actual)
        # In a full RL loop, you would use this reward to adjust your model.
        return reward

    def get_sentiment_history(self):
        """
        Return the sentiment counts over the conversation history.
        For visualization, we simply return the current counts.
        """
        return self.sentiment_counts.copy()

    def get_behavior_heatmap_data(self):
        """
        Create dummy heatmap data. For example, count the frequency of each sentiment by role.
        """
        roles = ["user", "assistant"]
        sentiments = ["positive", "neutral", "negative"]
        heatmap = np.zeros((len(roles), len(sentiments)))
        for entry in self.history:
            if entry["role"] in roles:
                row = roles.index(entry["role"])
                col = sentiments.index(entry["sentiment"])
                heatmap[row, col] += 1
        return roles, sentiments, heatmap

    def get_decision_flow_graph(self):
        """
        Build a simple decision flow graph using networkx, where each message is a node.
        """
        G = nx.DiGraph()
        for idx, entry in enumerate(self.history):
            label = f"{entry['role'][0].upper()}: {entry['message'][:10]}..."
            G.add_node(idx, label=label)
            if idx > 0:
                G.add_edge(idx - 1, idx)
        return G

    def get_embedding_projection(self):
        """
        Reduce the stored embeddings to 2D using PCA for visualization.
        """
        if not self.history:
            return np.empty((0, 2)), []
        embeddings = np.array([entry["embedding"] for entry in self.history])
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        roles = [entry["role"] for entry in self.history]
        return embeddings_2d, roles

