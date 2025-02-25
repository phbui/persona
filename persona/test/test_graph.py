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
    mg.close()

def test_full_hierarchy_and_retrieval(graph_manager):
    episodes = [
        {"content": "Alice visited New York last week and loved the food.", "timestamp": time.time()},
        {"content": "Bob went to New York and saw a Broadway show.", "timestamp": time.time()},
        {"content": "Charlie is planning to visit New York for a conference.", "timestamp": time.time()},
        {"content": "Diana discussed politics and art in New York with her colleagues.", "timestamp": time.time()},
        {"content": "Tinguan is Japanese but lives in China.", "timestamp": time.time()},
        {"content": "Eve traveled to Paris and enjoyed the art museums.", "timestamp": time.time()},
        {"content": "Frank attended a technology expo in San Francisco.", "timestamp": time.time()},
        {"content": "Grace participated in a local food festival in Rome.", "timestamp": time.time()},
        {"content": "Henry played football with friends in London.", "timestamp": time.time()},
        {"content": "Ivy wrote a book about the history of Berlin.", "timestamp": time.time()},
        {"content": "Jack explored the ancient ruins of Athens.", "timestamp": time.time()},
        {"content": "Karen attended a jazz festival in New Orleans.", "timestamp": time.time()},
        {"content": "Leo enjoyed the calm beaches of Bali.", "timestamp": time.time()},
        {"content": "Mona took a cooking class in Tokyo and learned new recipes.", "timestamp": time.time()},
        {"content": "Nate visited the bustling markets of Mumbai.", "timestamp": time.time()},
        {"content": "Olivia discovered hidden gems in Sydney.", "timestamp": time.time()},
        {"content": "Peter marveled at the skyscrapers of Dubai.", "timestamp": time.time()},
        {"content": "Quinn studied the ancient ruins in Cairo.", "timestamp": time.time()},
        {"content": "Rachel enjoyed a scenic drive through the Swiss Alps.", "timestamp": time.time()},
        {"content": "Steve experienced the vibrant culture of Rio de Janeiro.", "timestamp": time.time()},
    ]

    for episode in episodes:
        graph_manager.process_new_memory(episode, context_window=5)
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
