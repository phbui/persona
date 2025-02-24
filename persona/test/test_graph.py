import os
import sys
import time
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from manager.manager_graph import Manager_Graph

@pytest.fixture(scope="function", autouse=True)
def graph_manager():
    mg = Manager_Graph()
    mg.delete_entire_graph()
    yield mg
    mg.delete_entire_graph()
    mg.close()

def test_process_new_memory(graph_manager):
    episode_data = {
        "content": "Alice visited New York last week and loved the food.",
        "timestamp": time.time(),
        "embedding": [0.1, 0.2, 0.3]
    }
    graph_manager.process_new_memory(episode_data, context_window=4)
    episode_nodes = graph_manager.run_query("MATCH (e:Episode) RETURN e LIMIT 1")
    assert episode_nodes is not None and len(episode_nodes) > 0, "No Episode nodes found after processing memory."
    
def test_full_hierarchy_and_retrieval(graph_manager):
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
    graph_manager._update_entire_graph()
    community_nodes = graph_manager.run_query("MATCH (c:Community) RETURN c LIMIT 1")
    assert community_nodes is not None and len(community_nodes) > 0, "No Community nodes found in the graph."
    episode_nodes = graph_manager.run_query("MATCH (e:Episode) RETURN e LIMIT 1")
    assert episode_nodes is not None and len(episode_nodes) > 0, "No Episode nodes found in the graph."
    semantic_relationships = graph_manager.run_query("MATCH ()-[r:SEMANTICALLY_RELATED]->() RETURN r LIMIT 1")
    if semantic_relationships:
        assert len(semantic_relationships) > 0, "Semantic relationships should be present but none were found."
    query_text = "Tell me about New York"
    candidates = graph_manager.retrieve_candidates(query_text, result_limit=5)
    assert isinstance(candidates, list) and len(candidates) > 0, "No candidates retrieved for query."
    found = any("New York" in candidate["content"] for candidate in candidates)
    assert found, "Candidate matching 'New York' was not found."

if __name__ == "__main__":
    pytest.main()
