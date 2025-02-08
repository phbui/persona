import numpy as np
from sklearn.decomposition import PCA
import datetime
from transformers import pipeline

class PlayerModel:
    def __init__(self):
        self.history = []  
        self.sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        self.role_sentiment_counts = {
            "player": {"positive": 0, "neutral": 0, "negative": 0},
            "llm": {"positive": 0, "neutral": 0, "negative": 0}
        }
        
        self.emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            tokenizer="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
    
    def get_sentiment_emotion(self, text):
        """
        Uses the Emotion-English model pipeline to classify the text.
        Returns:
            predicted_emotion: The most likely emotion (e.g., "joy", "anger", etc.)
            score_dict: A dictionary containing the probabilities for each emotion and a computed compound score.
        """
        results = self.emotion_classifier(text)
        # results is a list (one per input) containing a list of dicts.
        emotion_scores = results[0]
        # Determine the predicted emotion as the one with highest score.
        predicted_emotion = max(emotion_scores, key=lambda x: x['score'])['label']
        # Build a dictionary of scores with lowercased keys.
        score_dict = {d['label'].lower(): d['score'] for d in emotion_scores}
        # Compute a compound score: compound = joy - average(anger, disgust, fear, sadness)
        neg_emotions = ['anger', 'disgust', 'fear', 'sadness']
        joy_prob = score_dict.get('joy', 0.0)
        negative_scores = [score_dict.get(emotion, 0.0) for emotion in neg_emotions]
        negative_avg = sum(negative_scores) / len(negative_scores) if negative_scores else 0.0
        compound = joy_prob - negative_avg
        score_dict['compound'] = compound
        
        return predicted_emotion, score_dict
    
    def update(self, embedding, chunked_embeddings, message, role):
        # Use the emotion pipeline for sentiment analysis.
        predicted_emotion, scores = self.get_sentiment_emotion(message)
        compound = scores["compound"]
        # Use thresholds to determine overall sentiment.
        if compound >= 0.05:
            sentiment = "positive"
        elif compound <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        # Update counts.
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
        Reduce an array of embeddings (shape: [m, d]) to n_components dimensions using PCA.
        If there's only one embedding, return the first n_components dimensions.
        Pads with zeros if the result has fewer than n_components columns.
        """
        embeddings = np.array(embeddings)
        m, _ = embeddings.shape
        effective_n_components = min(n_components, m)
        if m == 1:
            reduced = embeddings[:, :n_components]
        else:
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
