import json
import time
import numpy as np
import pytest
from manager.manager_extraction import Manager_Extraction

# Dummy responses for extraction tests.
VALID_ENTITY_RESPONSE_SIMPLE = json.dumps([
    {"content": "Alice", "embedding": None},
    {"content": "Paris", "embedding": None}
])
VALID_FACT_RESPONSE_SIMPLE = json.dumps([
    {"fact": "Alice visited Paris", "embedding": None},
    {"fact": "Bob is a teacher", "embedding": None}
])
# More complex responses:
VALID_ENTITY_RESPONSE_COMPLEX = json.dumps([
    {"content": "Alice", "embedding": [0.1, 0.2, 0.3]},
    {"content": "Paris", "embedding": [0.2, 0.3, 0.4]},
    {"content": "Bob", "embedding": [0.15, 0.25, 0.35]},
    {"content": "Google", "embedding": [0.3, 0.4, 0.5]}
])
VALID_FACT_RESPONSE_COMPLEX = json.dumps([
    {"fact": "Alice visited Paris", "embedding": [0.1, 0.1, 0.1]},
    {"fact": "Bob works at Google", "embedding": [0.2, 0.2, 0.2]}
])
INVALID_RESPONSE = ""  # Simulates an invalid (empty) response.

@pytest.fixture
def extraction_manager():
    mgr_extraction = Manager_Extraction()
    
    # By default, set generate_response to return a simple valid entity response.
    def dummy_generate_response(prompt, max_new_tokens=256, temperature=0.2):
        if "named entities" in prompt:
            return VALID_ENTITY_RESPONSE_SIMPLE
        elif "factual statements" in prompt:
            return VALID_FACT_RESPONSE_SIMPLE
        return INVALID_RESPONSE

    mgr_extraction.manager_llm.generate_response = dummy_generate_response
    
    # For resolve_entity, default to no candidates.
    mgr_extraction.run_query = lambda query, params: []
    
    return mgr_extraction

def test_extract_entities_success_simple(extraction_manager):
    text = "Alice visited Paris."
    entities = extraction_manager.extract_entities(text)
    print("Simple entity extraction output:", entities)
    assert isinstance(entities, list)
    assert len(entities) == 2
    assert entities[0]["content"] == "Alice"
    assert entities[1]["content"] == "Paris"

def test_extract_entities_success_complex(extraction_manager):
    # Override generate_response to return a more complex entity response.
    extraction_manager.manager_llm.generate_response = (
        lambda prompt, max_new_tokens, temperature: VALID_ENTITY_RESPONSE_COMPLEX
        if "named entities" in prompt else INVALID_RESPONSE
    )
    text = "Alice and Bob went to Paris and later visited Google headquarters."
    entities = extraction_manager.extract_entities(text)
    print("Complex entity extraction output:", entities)
    assert isinstance(entities, list)
    assert len(entities) == 4
    # Verify that each entity has a non-null embedding.
    for entity in entities:
        assert entity["embedding"] is not None, f"Entity {entity['content']} should have a non-null embedding."

def test_extract_entities_failure(extraction_manager):
    # Force an invalid response.
    extraction_manager.manager_llm.generate_response = (
        lambda prompt, max_new_tokens, temperature: INVALID_RESPONSE
    )
    text = "This text should yield an error."
    entities = extraction_manager.extract_entities(text)
    print("Entity extraction with invalid response:", entities)
    assert isinstance(entities, list)
    assert len(entities) == 0

def test_extract_facts_success_complex(extraction_manager):
    # Override generate_response to return a complex fact response.
    extraction_manager.manager_llm.generate_response = (
        lambda prompt, max_new_tokens, temperature: VALID_FACT_RESPONSE_COMPLEX
        if "factual statements" in prompt else INVALID_RESPONSE
    )
    text = "Alice visited Paris and Bob works at Google."
    facts = extraction_manager.extract_facts(text)
    print("Complex fact extraction output:", facts)
    assert isinstance(facts, list)
    assert len(facts) == 2
    assert facts[0]["fact"] == "Alice visited Paris"
    assert facts[1]["fact"] == "Bob works at Google"

def test_cosine_similarity(extraction_manager):
    vec1 = [1, 0, 0]
    vec2 = [1, 0, 0]
    vec3 = [0, 1, 0]
    sim_same = extraction_manager.cosine_similarity(vec1, vec2)
    sim_diff = extraction_manager.cosine_similarity(vec1, vec3)
    print(f"Cosine similarity (same): {sim_same}")
    print(f"Cosine similarity (different): {sim_diff}")
    assert np.isclose(sim_same, 1.0)
    assert np.isclose(sim_diff, 0.0)

def test_resolve_entity_no_candidates(extraction_manager):
    # With no candidates returned from run_query, resolve_entity should return the input.
    input_entity = {"content": "Alice", "embedding": [0.1, 0.2, 0.3]}
    resolved = extraction_manager.resolve_entity(input_entity)
    print("Resolved entity (no candidates):", resolved)
    assert resolved == input_entity

def test_resolve_entity_with_candidate(extraction_manager):
    # Simulate a candidate with a higher similarity.
    candidate_entity = {"content": "Alice Smith", "embedding": [0.1, 0.2, 0.3]}
    # Override run_query to return a candidate.
    extraction_manager.run_query = lambda query, params: [candidate_entity]
    # Override generate_response to simulate a "yes" answer for redundancy.
    extraction_manager.manager_llm.generate_response = (
        lambda prompt, max_new_tokens, temperature: "yes"
    )
    input_entity = {"content": "Alice", "embedding": [0.1, 0.2, 0.3]}
    resolved = extraction_manager.resolve_entity(input_entity)
    print("Resolved entity (with candidate):", resolved)
    # Since similarity is 1 and answer is "yes", we expect the candidate to be returned.
    assert resolved == candidate_entity

if __name__ == "__main__":
    pytest.main()
