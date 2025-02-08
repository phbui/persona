import copy
import numpy as np
from sklearn.decomposition import PCA
import torch
import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class PlayerModel:
    def __init__(self):
        self.history = []  
        self.sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        self.role_sentiment_counts = {
            "player": {"positive": 0, "neutral": 0, "negative": 0},
            "llm": {"positive": 0, "neutral": 0, "negative": 0}
        }
        self.twitter_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.twitter_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    
    
    def get_sentiment_twitter(self, text):
        tokens = self.twitter_tokenizer(text, return_tensors="pt")
        outputs = self.twitter_model(**tokens)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).tolist()[0]
        labels = ["negative", "neutral", "positive"]
        predicted_label = labels[probs.index(max(probs))]
        compound = probs[2] - probs[0]
        score_dict = {"neg": probs[0], "neu": probs[1], "pos": probs[2], "compound": compound}
        return predicted_label, score_dict
    
    def update(self, embedding, message, role):
        predicted_label, scores = self.get_sentiment_twitter(message)
        compound = scores["compound"]
        if compound >= 0.05:
            sentiment = "positive"
        elif compound <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        self.sentiment_counts[sentiment] += 1
        if role in self.role_sentiment_counts:
            self.role_sentiment_counts[role][sentiment] += 1
        else:
            self.role_sentiment_counts[role] = {"positive": 0, "neutral": 0, "negative": 0}
            self.role_sentiment_counts[role][sentiment] = 1
        order = len(self.history) + 1
        timestamp = datetime.datetime.now().isoformat()
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
        if not self.history:
            return []
        reduced_history = copy.deepcopy(self.history)
        embeddings = np.array([entry["embedding"] for entry in reduced_history])
        if len(embeddings) == 1:
            reduced_embeddings = embeddings[:, :3]
        elif len(embeddings) == 2:
            pca = PCA(n_components=2)
            reduced_2d = pca.fit_transform(embeddings)
            reduced_embeddings = np.hstack([reduced_2d, np.zeros((2, 1))])
        else:
            pca = PCA(n_components=3)
            reduced_embeddings = pca.fit_transform(embeddings)
        for i, entry in enumerate(reduced_history):
            entry["embedding"] = reduced_embeddings[i]
        return reduced_history
