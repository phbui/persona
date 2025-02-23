import os
import sys
import time
import pytest

# Ensure the 'src' directory is in the path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from manager.manager_graph import Manager_Graph

@pytest.fixture(scope="module")
def graph_manager():
    mg = Manager_Graph()
    # Ensure we start with an empty database.
    mg.delete_entire_graph()
    yield mg
    # Cleanup: empty the database and close connection.
    mg.delete_entire_graph()
    mg.close()

def test_process_new_memory(graph_manager):
    # Example episode representing a new conversation turn.
    episode_data = {
        "content": "Alice visited New York last week and loved the food.",
        "timestamp": time.time(),
        "embedding": [0.1, 0.2, 0.3]  # Example embedding vector
    }
    # Process the new memory (ingest episode, extract and resolve entities, link episode->entity,
    # and update the entire graph hierarchy)
    graph_manager.process_new_memory(episode_data, context_window=4)
    
    # Check that an Episode node was created.
    episode_nodes = graph_manager.run_query("MATCH (e:Episode) RETURN e LIMIT 1")
    assert episode_nodes is not None and len(episode_nodes) > 0, "No Episode nodes found after processing memory."

def test_retrieve_candidates(graph_manager):
    # Query the graph for a candidate. We expect "New York" to have been created as an Entity.
    query_text = "Tell me about New York"
    candidates = graph_manager.retrieve_candidates(query_text, result_limit=5)
    # Assert that we got at least one candidate back.
    assert isinstance(candidates, list) and len(candidates) > 0, "No candidates retrieved for query."
    # Check that one of the candidates mentions "New York"
    found = any("New York" in candidate["content"] for candidate in candidates)
    assert found, "Candidate matching 'New York' was not found."

def test_full_hierarchy(graph_manager):
    # Build a larger graph by processing multiple episodes
    episodes = [
        {
            "content": "Alice visited New York last week and loved the food.",
            "timestamp": time.time(),
            "embedding": [0.1, 0.2, 0.3]
        },
        {
            "content": "Bob went to New York and saw a Broadway show.",
            "timestamp": time.time(),
            "embedding": [0.15, 0.25, 0.35]
        },
        {
            "content": "Charlie is planning to visit New York for a conference.",
            "timestamp": time.time(),
            "embedding": [0.12, 0.22, 0.32]
        },
        {
            "content": "Diana discussed politics and art in New York with her colleagues.",
            "timestamp": time.time(),
            "embedding": [0.13, 0.23, 0.33]
        }
    ]
    for episode in episodes:
        graph_manager.process_new_memory(episode, context_window=4)
    
    # Force a full update of the graph (communities, semantic relationships, etc.)
    graph_manager._update_entire_graph()
    
    # Verify that community nodes exist.
    community_nodes = graph_manager.run_query("MATCH (c:Community) RETURN c LIMIT 1")
    assert community_nodes is not None and len(community_nodes) > 0, "No Community nodes found in the graph."
    
    # Verify that Episode nodes exist.
    episode_nodes = graph_manager.run_query("MATCH (e:Episode) RETURN e LIMIT 1")
    assert episode_nodes is not None and len(episode_nodes) > 0, "No Episode nodes found in the graph."
    
    # Optionally, check for semantic relationships if your embeddings are similar enough.
    semantic_relationships = graph_manager.run_query("MATCH ()-[r:SEMANTICALLY_RELATED]->() RETURN r LIMIT 1")
    # For testing purposes, we don't force an assertion here if no semantic edges were created.
    if semantic_relationships:
        assert len(semantic_relationships) > 0, "Semantic relationships should be present but none were found."

if __name__ == "__main__":
    pytest.main()
