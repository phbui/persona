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
    
    def describe_face(self, au_intensities):
        intensity_descriptions = {
            1: "slight",
            2: "moderate",
            3: "strong"
        }

        emotion_aus = {
            'happiness': [6, 12],
            'sadness': [1, 4, 15],
            'surprise': [1, 2, 5, 26],
            'fear': [1, 2, 4, 5, 20, 26],
            'anger': [4, 5, 7, 23],
            'disgust': [9, 15, 16]
        }

        au_index_map = {au: idx for idx, au in enumerate([1, 2, 4, 5, 6, 7, 9, 11, 12, 14, 15, 17, 20, 23, 24, 25, 26, 28, 43])}

        emotion_intensity = []
        
        for emotion, aus in emotion_aus.items():
            intensities = [au_intensities[au_index_map[au]] for au in aus if au in au_index_map and au_intensities[au_index_map[au]] > 0]
            if intensities:
                min_intensity = min(intensities)
                emotion_intensity.append((min_intensity, emotion))

        emotion_intensity.sort(reverse=True, key=lambda x: x[0]) 

        descriptions = [f"{intensity_descriptions.get(intensity, 'unknown')} {emotion}" for intensity, emotion in emotion_intensity]

        if descriptions:
            return f"The face displays {', '.join(descriptions)}."
        else:
            return "The face displays a neutral expression."