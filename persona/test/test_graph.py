import os
import pytest
from manager.manager_graph import Manager_Graph

@pytest.fixture(scope="module")
def graph_manager():
    # Setup: Instantiate the graph manager
    mg = Manager_Graph()
    yield mg
    # Teardown: Clear the graph and close the connection
    mg.delete_entire_graph()
    mg.close()

def test_add_nodes(graph_manager):
    nodes = [
        {"label": "TreeNode", "properties": {"name": "root"}},
        {"label": "TreeNode", "properties": {"name": "child1"}},
        {"label": "TreeNode", "properties": {"name": "child2"}},
        {"label": "TreeNode", "properties": {"name": "child3"}}
    ]
    graph_manager.add_nodes(nodes)
    result = graph_manager.search_nodes_by_property("TreeNode", "name", "root")
    assert len(result) == 1
    assert result[0]['n']['name'] == 'root'

def test_add_subgraph(graph_manager):
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
    result = graph_manager.run_query(
        "MATCH (n:TreeNode {name: 'root'})-[:HAS_CHILD]->(c) RETURN c"
    )
    assert len(result) == 3

def test_download_and_upload_graph(graph_manager, tmpdir):
    download_path = tmpdir.mkdir("downloads")
    filename = "test_graph_download.json"
    success = graph_manager.download_entire_graph(str(download_path), filename)
    assert success
    assert os.path.isfile(os.path.join(download_path, filename))

    # Clear the graph and upload from the downloaded file
    graph_manager.delete_entire_graph()
    upload_success = graph_manager.upload_graph_from_json(os.path.join(download_path, filename))
    assert upload_success
    result = graph_manager.run_query("MATCH (n:TreeNode) RETURN n")
    assert len(result) == 4  # root + 3 children

def test_graph_search(graph_manager):
    root_id_result = graph_manager.run_query("MATCH (n:TreeNode {name: 'root'}) RETURN id(n) AS id")
    assert root_id_result
    root_id = root_id_result[0]["id"]
    result = graph_manager.graph_search(root_id, 2)
    assert 'nodes' in result
    assert 'relationships' in result
    assert len(result['nodes']) == 4
    assert len(result['relationships']) == 3
