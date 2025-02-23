import json
import time
import numpy as np
import pytest
from manager.manager_extraction import Manager_Extraction

@pytest.fixture
def extraction_manager():
    # Create an instance of Manager_Extraction.
    mgr_extraction = Manager_Extraction()
    # Do not override generate_response; use the actual model responses.
    # Also, for resolve_entity, we let run_query be the default (which may return an empty list).
    return mgr_extraction

def test_extract_entities_real(extraction_manager):
    text = "Japan was nuked by North Korea."
    # This will send the prompt to the LLM and then generate embeddings using the model.
    entities = extraction_manager.extract_entities(text)
    print("Extracted entities:", entities)
    # Check that the returned value is a list.
    assert isinstance(entities, list), "Output is not a list."
    # We expect at least one entity to be extracted.
    assert len(entities) > 0, "No entities extracted."
    # Each entity should have a 'content' key and a computed embedding.
    for entity in entities:
        assert "content" in entity, "Entity missing 'content' key."
        assert "embedding" in entity, f"Entity '{entity['content']}' missing 'embedding' key."
        # Check that the embedding is a non-empty list of numbers.
        embedding = entity["embedding"]
        assert isinstance(embedding, list) and len(embedding) > 0, f"Invalid embedding for entity '{entity['content']}'."
        # Optionally, print embedding length and a snippet of values.
        print(f"Entity '{entity['content']}' embedding (first 5 values): {embedding[:5]}")

def test_extract_facts_real(extraction_manager):
    text = "John is a farmer, but Jake is a war criminal."
    facts = extraction_manager.extract_facts(text)
    print("Extracted facts:", facts)
    assert isinstance(facts, list), "Output is not a list."
    assert len(facts) > 0, "No facts extracted."
    for fact in facts:
        assert "fact" in fact, "Fact missing 'fact' key."
        assert "embedding" in fact, f"Fact '{fact['fact']}' missing 'embedding' key."
        embedding = fact["embedding"]
        assert isinstance(embedding, list) and len(embedding) > 0, f"Invalid embedding for fact '{fact['fact']}'."
        print(f"Fact '{fact['fact']}' embedding (first 5 values): {embedding[:5]}")

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
