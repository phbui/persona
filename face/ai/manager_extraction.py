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
        # Mapping of intensity levels to descriptive terms
        intensity_descriptions = {
            0: "no sign of",
            1: "slight",
            2: "moderate",
            3: "strong"
        }

        # Mapping of emotions to their corresponding AUs
        emotion_aus = {
            'happiness': [6, 12],
            'sadness': [1, 4, 15],
            'surprise': [1, 2, 5, 26],
            'fear': [1, 2, 4, 5, 20, 26],
            'anger': [4, 5, 7, 23],
            'disgust': [9, 15, 16]
        }

        # Mapping of AU numbers to their descriptions
        au_descriptions = {
            1: "inner brow raiser",
            2: "outer brow raiser",
            4: "brow lowerer",
            5: "upper lid raiser",
            6: "cheek raiser",
            7: "lid tightener",
            9: "nose wrinkler",
            12: "lip corner puller",
            15: "lip corner depressor",
            16: "lower lip depressor",
            20: "lip stretcher",
            23: "lip tightener",
            26: "jaw drop",
            43: "eyes closed"
        }

        # Initialize a list to hold descriptions of active AUs
        active_au_descriptions = []

        # Iterate over each AU and its intensity
        for au, intensity in enumerate(au_intensities, start=1):
            if intensity > 0 and au in au_descriptions:
                # Get the descriptive term for the intensity
                intensity_desc = intensity_descriptions.get(intensity, 'unknown')
                # Get the description of the AU
                au_desc = au_descriptions[au]
                # Append the formatted description to the list
                active_au_descriptions.append(f"{intensity_desc} {au_desc}")

        # Join all AU descriptions into a single string
        au_description_str = "; ".join(active_au_descriptions)

        # Initialize a list to hold descriptions of emotions
        emotion_intensity = []

        # Iterate over each emotion and its corresponding AUs
        for emotion, aus in emotion_aus.items():
            # Get the intensities of the AUs related to the emotion
            intensities = [au_intensities[au - 1] for au in aus if au - 1 < len(au_intensities)]
            if intensities:
                # Determine the minimum intensity among the relevant AUs
                min_intensity = min(intensities)
                # Append the intensity and emotion to the list
                emotion_intensity.append((min_intensity, emotion))

        # Sort emotions by intensity in descending order
        emotion_intensity.sort(reverse=True, key=lambda x: x[0])

        # Create a list of descriptions for each emotion based on its intensity
        emotion_descriptions = [f"{intensity_descriptions.get(intensity, 'unknown')} {emotion}" for intensity, emotion in emotion_intensity]

        # Combine emotion descriptions into a single string
        emotion_description_str = ", ".join(emotion_descriptions)

        # Construct the final description
        if emotion_descriptions:
            return f"The face displays {emotion_description_str}. Specific movements: {au_description_str}."
        else:
            return "The face shows no discernible emotions."
