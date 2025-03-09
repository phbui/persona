import json
import numpy as np
from ai.manager_analysis import Manager_Analysis

class Manager_Extraction:
    def __init__(self):
        self.manage_analysis_sentiment = Manager_Analysis(
            "sentiment-analysis", "siebert/sentiment-roberta-large-english"
        )
        self.manage_analysis_emotion = Manager_Analysis(
            "text-classification", "j-hartmann/emotion-english-distilroberta-base"
        )

    def extract_features(self, text):
        """Extracts a 9D feature vector (sentiment + emotion scores)."""
        sentiment_result = self.manage_analysis_sentiment.analyze(text)
        positive_score = 0
        negative_score = 0

        if sentiment_result:
            label = sentiment_result[0]["label"]
            score = sentiment_result[0]["score"]
            if label == "POSITIVE":
                positive_score = score
                negative_score = 1 - score
            elif label == "NEGATIVE":
                negative_score = score
                positive_score = 1 - score

        emotion_result = self.manage_analysis_emotion.analyze(text, top_k=None)

        target_emotions = ["disgust", "neutral", "anger", "joy", "surprise", "fear", "sadness"]
        emotion_scores = {e["label"]: e["score"] for e in emotion_result}

        emotion_vector = np.array([emotion_scores.get(em, 0) for em in target_emotions])
        feature_vector = np.concatenate(([positive_score, negative_score], emotion_vector))

        return feature_vector  # Shape: (9,)
