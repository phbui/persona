import os
import sys
import numpy as np
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from manager.ai.manager_extraction import Manager_Extraction

@pytest.fixture
def extraction_manager():
    # Create an instance of Manager_Extraction.
    mgr_extraction = Manager_Extraction()
    # Do not override generate_response; use the actual model responses.
    # Also, for resolve_entity, we let run_query be the default (which may return an empty list).
    return mgr_extraction

def test_extract_entities_real(extraction_manager):
    text = "Japan was nuked by North Korea."
    
    # Extract entities (should also handle sentiment & emotion now)
    entities = extraction_manager.extract_entities(text)
    print("Extracted entities:", entities)

    assert isinstance(entities, list), "Output is not a list."
    assert len(entities) > 0, "No entities extracted."

    for entity in entities:
        assert "content" in entity, "Entity missing 'content' key."
        assert "embedding" in entity, f"Entity '{entity['content']}' missing 'embedding' key."
        
        # Check embedding validity
        embedding = entity["embedding"]
        assert isinstance(embedding, list) and len(embedding) > 0, f"Invalid embedding for entity '{entity['content']}'."

        # Validate sentiment
        assert "sentiment" in entity, f"Entity '{entity['content']}' missing 'sentiment' key."
        sentiment = entity["sentiment"]
        assert isinstance(sentiment, list), f"Sentiment must be a list, got {type(sentiment)}."
        assert len(sentiment) > 0, "Sentiment list is empty."
        assert isinstance(sentiment[0], dict), f"Sentiment list must contain a dictionary, got {type(sentiment[0])}."
        assert "label" in sentiment[0] and "score" in sentiment[0], "Sentiment dictionary missing 'label' or 'score' key."
        assert isinstance(sentiment[0]["label"], str), "Sentiment label must be a string."
        assert isinstance(sentiment[0]["score"], float), "Sentiment score must be a float."
        assert 0 <= sentiment[0]["score"] <= 1, f"Sentiment score out of range: {sentiment[0]['score']}."
        
        # Validate emotion
        assert "emotion" in entity, f"Entity '{entity['content']}' missing 'emotion' key."
        emotion = entity["emotion"][0]
        assert isinstance(emotion, list), f"Emotion must be a list, got {type(emotion)}."
        assert len(emotion) > 0, "Emotion list is empty."
        assert isinstance(emotion[0], dict), f"Emotion list must contain a dictionary, got {type(emotion[0])}."
        assert "label" in emotion[0] and "score" in emotion[0], "Emotion dictionary missing 'label' or 'score' key."
        assert isinstance(emotion[0]["label"], str), "Emotion label must be a string."
        assert isinstance(emotion[0]["score"], float), "Emotion score must be a float."
        assert 0 <= emotion[0]["score"] <= 1, f"Emotion score out of range: {emotion[0]['score']}."

        print(f"Entity '{entity['content']}' - Sentiment: {sentiment}, Emotions: {emotion}")

def test_cosine_similarity(extraction_manager):
    vec1 = [1, 0, 0]
    vec2 = [1, 0, 0]
    vec3 = [0, 1, 0]
    sim_same = extraction_manager.cosine_similarity(vec1, vec2)
    sim_diff = extraction_manager.cosine_similarity(vec1, vec3)
    print(f"Cosine similarity (same vectors): {sim_same}")
    print(f"Cosine similarity (different vectors): {sim_diff}")
    assert np.isclose(sim_same, 1.0), "Cosine similarity for identical vectors should be 1.0."
    assert np.isclose(sim_diff, 0.0), "Cosine similarity for orthogonal vectors should be 0.0."

if __name__ == "__main__":
    pytest.main()
