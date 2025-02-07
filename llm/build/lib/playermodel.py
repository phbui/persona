import copy
import numpy as np
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import datetime

class PlayerModel:
    def __init__(self, embedding_dim=384):
        # Initialize conversation history.
        # Each history entry is a dict with keys:
        # 'order', 'timestamp', 'role', 'message', 'embedding', 'sentiment', 'sentiment_scores'.
        self.history = []  
        
        # Overall sentiment counts.
        self.sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        
        # Sentiment counts separated by role.
        self.role_sentiment_counts = {
            "player": {"positive": 0, "neutral": 0, "negative": 0},
            "llm": {"positive": 0, "neutral": 0, "negative": 0}
        }
        
        self.embedding_dim = embedding_dim
        
        # Use SentenceTransformer to generate semantic embeddings.
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Set up the Twitter RoBERTa sentiment analyzer.
        # This model is from Cardiff NLP and is trained on Twitter data.
        self.twitter_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.twitter_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    
    def get_embedding(self, text):
        """
        Generate a high-quality embedding from the text using SentenceTransformer.
        Returns a 384-dimensional vector.
        """
        return self.encoder.encode(text)
    
    def get_sentiment_twitter(self, text):
        """
        Use the Twitter RoBERTa model to analyze sentiment.
        Returns a tuple: (predicted_label, score_dict)
        
        score_dict contains:
          - 'neg': Negative probability
          - 'neu': Neutral probability
          - 'pos': Positive probability
          - 'compound': (pos - neg), as a rough compound score
        """
        # Tokenize and get model outputs
        tokens = self.twitter_tokenizer(text, return_tensors="pt")
        outputs = self.twitter_model(**tokens)
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).tolist()[0]
        # The Cardiff model's labels are (in order): Negative, Neutral, Positive.
        labels = ["negative", "neutral", "positive"]
        predicted_label = labels[probs.index(max(probs))]
        # Create a compound score: positive probability minus negative probability.
        compound = probs[2] - probs[0]
        score_dict = {
            "neg": probs[0],
            "neu": probs[1],
            "pos": probs[2],
            "compound": compound
        }
        return predicted_label, score_dict
    
    def update(self, message, role):
        """
        Update the model with a new message.
        
        This method:
          - Computes a semantic embedding for the message.
          - Runs context-aware sentiment analysis using the Twitter RoBERTa model to get scores.
          - Determines a sentiment label based on the compound score 
            (compound >= 0.05 → positive, compound <= -0.05 → negative, otherwise neutral).
          - Updates both overall sentiment counts and counts by role.
          - Records the order of the message and a timestamp.
          - Stores the message, its embedding, sentiment label, the full sentiment score dictionary,
            the order, and timestamp in the conversation history.
        """
        # Compute the semantic embedding.
        embedding = self.get_embedding(message)
        
        # Analyze sentiment using Twitter RoBERTa.
        predicted_label, scores = self.get_sentiment_twitter(message)
        compound = scores["compound"]
        
        # Determine sentiment based on compound score thresholds.
        if compound >= 0.05:
            sentiment = "positive"
        elif compound <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        # Update overall sentiment counts.
        self.sentiment_counts[sentiment] += 1
        
        # Update sentiment counts by role.
        if role in self.role_sentiment_counts:
            self.role_sentiment_counts[role][sentiment] += 1
        else:
            self.role_sentiment_counts[role] = {"positive": 0, "neutral": 0, "negative": 0}
            self.role_sentiment_counts[role][sentiment] = 1
        
        # Record the order (i.e. the index in the conversation) and the current timestamp.
        order = len(self.history) + 1
        timestamp = datetime.datetime.now().isoformat()
        
        # Append the new data to the conversation history.
        self.history.append({
            "order": order,
            "timestamp": timestamp,
            "role": role,
            "message": message,
            "embedding": embedding,
            "sentiment": sentiment,
            "sentiment_scores": scores,
            "role_sentiment_counts": self.role_sentiment_counts[role],
            "overall_sentiment_counts": self.sentiment_counts
        })

    def get_history_for_visualization(self):
        """
        Reduces the 384-dimensional embeddings to 3D using PCA.
        Returns a **copy** of the history with updated 3D embeddings.
        """
        if not self.history:
            return []  # Return an empty list if no history exists.

        # Copy the history to avoid modifying the original data.
        reduced_history = copy.deepcopy(self.history)

        # Extract all embeddings into a matrix (num_messages, 384).
        embeddings = np.array([entry["embedding"] for entry in reduced_history])

        if len(embeddings) == 1:
            reduced_embeddings = embeddings[:, :3]  # Take first 3 dimensions

        elif len(embeddings) == 2:
            pca = PCA(n_components=2)
            reduced_2d = pca.fit_transform(embeddings)
            reduced_embeddings = np.hstack([reduced_2d, np.zeros((2, 1))])  # Pad to 3D

        else:
            pca = PCA(n_components=3)
            reduced_embeddings = pca.fit_transform(embeddings)

        for i, entry in enumerate(reduced_history):
            entry["embedding"] = reduced_embeddings[i]

        return reduced_history
