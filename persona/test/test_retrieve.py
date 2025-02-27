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

def test_retrieval(graph_manager):
    query_text = "Tell me about the battles."
    candidates = graph_manager.retrieve_candidates(query_text, result_limit=10)
    
    print(query_text)

    for candidate in candidates:
        print("\n")
        print(candidate)


if __name__ == "__main__":
    pytest.main()
