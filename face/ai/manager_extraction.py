import numpy as np
from ai.manager_analysis import Manager_Analysis
from ai.manager_encoder import Manager_Encoder

class Manager_Extraction:
    def __init__(self):
        self.manager_analysis_sentiment = Manager_Analysis(
            "sentiment-analysis", "siebert/sentiment-roberta-large-english"
        )
        self.manager_analysis_emotion = Manager_Analysis(
            "text-classification", "j-hartmann/emotion-english-distilroberta-base"
        )
        self.manager_encoder = Manager_Encoder("google/flan-t5-base")

        # Define the fixed embedding size based on the model configuration
        self.embedding_size = self.manager_encoder.model.config.d_model  # Typically 768D for Flan-T5 Base

    def extract_features(self, text):
        """Extracts a fixed-size feature vector (embedding + sentiment + emotion scores).
        """
        
        text_embedding = self.manager_encoder.generate_embedding(text)
        if len(text_embedding) != self.embedding_size:
            text_embedding = np.zeros(self.embedding_size)
        else:
            text_embedding = np.array(text_embedding) 

        sentiment_result = self.manager_analysis_sentiment.analyze(text)
        positive_score = 0.0
        negative_score = 0.0

        if sentiment_result:
            label = sentiment_result[0]["label"]
            score = sentiment_result[0]["score"]
            if label == "POSITIVE":
                positive_score = score
                negative_score = 1.0 - score
            elif label == "NEGATIVE":
                negative_score = score
                positive_score = 1.0 - score


        emotion_result = self.manager_analysis_emotion.analyze(text, top_k=None)
        target_emotions = ["disgust", "neutral", "anger", "joy", "surprise", "fear", "sadness"]
        emotion_scores = {e["label"]: e["score"] for e in emotion_result}
        emotion_vector = np.array([emotion_scores.get(em, 0) for em in target_emotions])

        feature_vector = np.concatenate((text_embedding, [positive_score, negative_score], emotion_vector))

        return feature_vector  # Shape: (embedding_size + 9,)

    def describe_face(au_intensities):
        # Mapping of intensity levels to descriptive terms
        intensity_descriptions = {
            0: "no activation of",
            1: "slight activation of",
            2: "moderate activation of",
            3: "strong activation of"
        }

        # Mapping of AU numbers to their muscle movement descriptions
        au_descriptions = {
            1: "inner brow raiser",
            2: "outer brow raiser",
            4: "brow lowerer",
            5: "upper eyelid raiser",
            6: "cheek raiser",
            7: "eyelid tightener",
            9: "nose wrinkler",
            12: "lip corner puller",
            15: "lip corner depressor",
            16: "lower lip depressor",
            20: "lip stretcher",
            23: "lip tightener",
            26: "jaw drop",
            43: "eyes closed"
        }

        # Build AU-based physical description
        movement_descriptions = []
        for au, intensity in enumerate(au_intensities, start=1):
            if au in au_descriptions:
                desc = intensity_descriptions.get(intensity, "unknown intensity")
                movement = au_descriptions[au]
                movement_descriptions.append(f"{desc} the {movement}")

        if not movement_descriptions:
            return "The face shows no visible muscle movement."
        return "Facial movement includes: " + "; ".join(movement_descriptions) + "."

