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
    
    def update(self, embedding, chunked_embeddings, message, role):
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
            "chunked_embeddings": chunked_embeddings,
            "sentiment": sentiment,
            "sentiment_scores": scores,
            "role_sentiment_counts": self.role_sentiment_counts[role],
            "overall_sentiment_counts": self.sentiment_counts
        })

    def reduce_array(self, embeddings, n_components=3):
        """
        Reduce a single array of embeddings (shape: [m, d]) to n_components dimensions using PCA.
        If there's only one embedding, it simply returns the first n_components dimensions.
        Pads with zeros if the result has fewer than n_components columns.
        
        Parameters:
        embeddings (array-like): An array of shape (m, d).
        n_components (int): Target number of dimensions.
        
        Returns:
        np.ndarray: Reduced embeddings with shape (m, n_components).
        """
        embeddings = np.array(embeddings)
        m, _ = embeddings.shape
        effective_n_components = min(n_components, m)
        if m == 1:
            reduced = embeddings[:, :n_components]
        else:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=effective_n_components)
            reduced = pca.fit_transform(embeddings)
            if reduced.shape[1] < n_components:
                pad_width = n_components - reduced.shape[1]
                reduced = np.hstack([reduced, np.zeros((m, pad_width))])
        return reduced

    def reduce_entry_embeddings(self, main_embedding, chunked_embeddings, n_components=3):
        """
        For a single history entry, combine the main embedding and its chunked embeddings,
        then compute PCA on the union using the helper function. Return the reduced main embedding
        and the reduced chunk embeddings.
        """
        combined = [main_embedding]
        if chunked_embeddings is not None:
            combined.extend(chunked_embeddings)
        combined = np.array(combined)
        reduced = self.reduce_array(combined, n_components)
        new_main = reduced[0]
        new_chunks = reduced[1:] if reduced.shape[0] > 1 else None
        return new_main, new_chunks


    def get_history_for_visualization(self):
        import copy
        if not self.history:
            return []
        reduced_history = copy.deepcopy(self.history)

        # For each history entry, reduce the main and chunked embeddings on a per-message basis.
        for entry in reduced_history:
            main_emb = entry["embedding"]
            chunked_embs = entry.get("chunked_embeddings", None)
            new_main, new_chunks = self.reduce_entry_embeddings(main_emb, chunked_embs, n_components=3)
            entry["embedding"] = new_main
            entry["chunked_embeddings"] = new_chunks

        return reduced_history
