import os
import sys
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

def test_add_nodes(graph_manager):
    graph_manager.delete_entire_graph()
    nodes = [
        {"label": "TreeNode", "properties": {"name": "root"}},
        {"label": "TreeNode", "properties": {"name": "child1"}},
        {"label": "TreeNode", "properties": {"name": "child2"}},
        {"label": "TreeNode", "properties": {"name": "child3"}}
    ]
    graph_manager.add_nodes(nodes)
    result = graph_manager.search_nodes_by_property("TreeNode", "name", "root")
    assert result is not None and len(result) == 1

def test_add_subgraph(graph_manager):
    graph_manager.delete_entire_graph()
    subgraph = {
        "nodes": [
            {"label": "TreeNode", "properties": {"name": "root"}},
            {"label": "TreeNode", "properties": {"name": "child1"}},
            {"label": "TreeNode", "properties": {"name": "child2"}},
            {"label": "TreeNode", "properties": {"name": "child3"}}
        ],
        "relationships": [
            {
                "start": {"label": "TreeNode", "match_properties": {"name": "root"}},
                "end": {"label": "TreeNode", "match_properties": {"name": "child1"}},
                "type": "HAS_CHILD",
                "properties": {}
            },
            {
                "start": {"label": "TreeNode", "match_properties": {"name": "root"}},
                "end": {"label": "TreeNode", "match_properties": {"name": "child2"}},
                "type": "HAS_CHILD",
                "properties": {}
            },
            {
                "start": {"label": "TreeNode", "match_properties": {"name": "root"}},
                "end": {"label": "TreeNode", "match_properties": {"name": "child3"}},
                "type": "HAS_CHILD",
                "properties": {}
            }
        ]
    }
    graph_manager.add_subgraph(subgraph)
    result = graph_manager.run_query("MATCH (n:TreeNode {name: 'root'})-[:HAS_CHILD]->(c) RETURN c")
    # Expect exactly 3 child relationships from the root.
    assert result is not None and len(result) == 3

def test_download_and_upload_graph(graph_manager, tmp_path):
    graph_manager.delete_entire_graph()
    # Build a tree structure.
    nodes = [
        {"label": "TreeNode", "properties": {"name": "root"}},
        {"label": "TreeNode", "properties": {"name": "child1"}},
        {"label": "TreeNode", "properties": {"name": "child2"}},
        {"label": "TreeNode", "properties": {"name": "child3"}}
    ]
    graph_manager.add_nodes(nodes)
    subgraph = {
        "nodes": nodes,
        "relationships": [
            {
                "start": {"label": "TreeNode", "match_properties": {"name": "root"}},
                "end": {"label": "TreeNode", "match_properties": {"name": "child1"}},
                "type": "HAS_CHILD",
                "properties": {}
            },
            {
                "start": {"label": "TreeNode", "match_properties": {"name": "root"}},
                "end": {"label": "TreeNode", "match_properties": {"name": "child2"}},
                "type": "HAS_CHILD",
                "properties": {}
            },
            {
                "start": {"label": "TreeNode", "match_properties": {"name": "root"}},
                "end": {"label": "TreeNode", "match_properties": {"name": "child3"}},
                "type": "HAS_CHILD",
                "properties": {}
            }
        ]
    }
    graph_manager.add_subgraph(subgraph)

    download_file = tmp_path / "test_graph_download.json"
    success = graph_manager.download_entire_graph(str(tmp_path), download_file.name)
    assert success
    assert download_file.is_file()

    # Clear graph and upload from downloaded file.
    graph_manager.delete_entire_graph()
    upload_success = graph_manager.upload_graph_from_json(str(download_file))
    assert upload_success
    # Expect 4 nodes (assuming unique nodes with label "TreeNode" for root and three children).
    result = graph_manager.run_query("MATCH (n:TreeNode) RETURN n")
    assert result is not None and len(result) == 4

def test_graph_search(graph_manager):
    graph_manager.delete_entire_graph()
    nodes = [
        {"label": "TreeNode", "properties": {"name": "root"}},
        {"label": "TreeNode", "properties": {"name": "child1"}},
        {"label": "TreeNode", "properties": {"name": "child2"}},
        {"label": "TreeNode", "properties": {"name": "child3"}}
    ]
    graph_manager.add_nodes(nodes)
    subgraph = {
        "nodes": nodes,
        "relationships": [
            {
                "start": {"label": "TreeNode", "match_properties": {"name": "root"}},
                "end": {"label": "TreeNode", "match_properties": {"name": "child1"}},
                "type": "HAS_CHILD",
                "properties": {}
            },
            {
                "start": {"label": "TreeNode", "match_properties": {"name": "root"}},
                "end": {"label": "TreeNode", "match_properties": {"name": "child2"}},
                "type": "HAS_CHILD",
                "properties": {}
            },
            {
                "start": {"label": "TreeNode", "match_properties": {"name": "root"}},
                "end": {"label": "TreeNode", "match_properties": {"name": "child3"}},
                "type": "HAS_CHILD",
                "properties": {}
            }
        ]
    }
    graph_manager.add_subgraph(subgraph)
    root_result = graph_manager.run_query("MATCH (n:TreeNode {name: 'root'}) RETURN id(n) AS id")
    assert root_result and len(root_result) > 0
    root_id = root_result[0]["id"]
    search_result = graph_manager.graph_search(root_id, 2)
    # The graph_search result is a list of records; extract the first record.
    assert search_result is not None and isinstance(search_result, list)
    record = search_result[0]
    assert "nodes" in record and len(record["nodes"]) >= 4

if __name__ == "__main__":
    pytest.main()
