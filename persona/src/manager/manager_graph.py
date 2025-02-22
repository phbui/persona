import os
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv
from meta.meta_singleton import Meta_Singleton
from log.logger import Logger, Log
from manager.manager_file import Manager_File

load_dotenv()
neo4js_uri = os.getenv('NEO4J_URI')
neo4js_user = os.getenv('NEO4J_USERNAME')
neo4js_password = os.getenv('NEO4J_PASSWORD')

class Manager_Graph(metaclass=Meta_Singleton):
    def __init__(self):
        self.uri = neo4js_uri
        self.user = neo4js_user
        self.password = neo4js_password
        self.logger = Logger()
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            log = Log("INFO", "graph", self.__class__.__name__, "__init__", f"Connected to Neo4j instance at {self.uri}")
            self.logger.add_log_obj(log)
        except Exception as e:
            log = Log("ERROR", "graph", self.__class__.__name__, "__init__", f"Failed to connect to Neo4j at {self.uri}: {e}")
            self.logger.add_log_obj(log)
            raise

    def close(self):
        self.driver.close()
        log = Log("INFO", "graph", self.__class__.__name__, "close", "Connection closed.")
        self.logger.add_log_obj(log)

    def run_query(self, query, parameters=None):
        if parameters is None:
            parameters = {}
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters)
                data = [record.data() for record in result]
                log = Log("INFO", "graph", self.__class__.__name__, "run_query", f"Executed query: {query} with parameters: {parameters}")
                self.logger.add_log_obj(log)
                return data
        except Exception as e:
            log = Log("ERROR", "graph", self.__class__.__name__, "run_query", f"Query failed: {query} with error: {e}")
            self.logger.add_log_obj(log)
            return None

    def add_nodes(self, nodes):
        for node in nodes:
            label = node.get('label')
            properties = node.get('properties', {})
            query = f"CREATE (n:{label} $properties)"
            self.run_query(query, {"properties": properties})
        log = Log("INFO", "graph", self.__class__.__name__, "add_nodes", f"Added {len(nodes)} nodes to the graph.")
        self.logger.add_log_obj(log)

    def add_subgraph(self, subgraph):
        nodes = subgraph.get('nodes', [])
        relationships = subgraph.get('relationships', [])
        if nodes:
            self.add_nodes(nodes)
        for rel in relationships:
            start = rel.get('start')
            end = rel.get('end')
            rel_type = rel.get('type')
            properties = rel.get('properties', {})
            start_label = start.get('label')
            start_props = start.get('match_properties')
            end_label = end.get('label')
            end_props = end.get('match_properties')
            query = f"MATCH (a:{start_label} $start_props), (b:{end_label} $end_props) CREATE (a)-[r:{rel_type} $properties]->(b)"
            self.run_query(query, {"start_props": start_props, "end_props": end_props, "properties": properties})
        log = Log("INFO", "graph", self.__class__.__name__, "add_subgraph", f"Added subgraph with {len(nodes)} nodes and {len(relationships)} relationships.")
        self.logger.add_log_obj(log)

    def delete_entire_graph(self):
        query = "MATCH (n) DETACH DELETE n"
        self.run_query(query)
        log = Log("INFO", "graph", self.__class__.__name__, "delete_entire_graph", "Deleted the entire graph.")
        self.logger.add_log_obj(log)

    def download_entire_graph(self, dir_path, filename="graph_download.json"):
        nodes_query = "MATCH (n) RETURN n"
        nodes_result = self.run_query(nodes_query)
        rel_query = "MATCH ()-[r]->() RETURN r"
        rel_result = self.run_query(rel_query)
        graph_data = {"nodes": nodes_result if nodes_result is not None else [], "relationships": rel_result if rel_result is not None else []}
        file_manager = Manager_File()
        success = file_manager.download_file(graph_data, dir_path, filename)
        log = Log("INFO", "graph", self.__class__.__name__, "download_entire_graph", f"Downloaded the entire graph as JSON to {os.path.join(dir_path, filename)}")
        self.logger.add_log_obj(log)
        return success

    @staticmethod
    def convert_json_to_cypher(graph_data):
        queries = []
        for node in graph_data.get('nodes', []):
            node_props = node.get('n')
            if node_props:
                query = "CREATE (n:Node $props)"
                queries.append((query, {"props": node_props}))
        for rel in graph_data.get('relationships', []):
            rel_props = rel.get('r')
            if rel_props:
                query = "MATCH (a:Node), (b:Node) WHERE id(a) = $id_a AND id(b) = $id_b CREATE (a)-[r:REL $props]->(b)"
                queries.append((query, {"id_a": rel_props.get('id_a'), "id_b": rel_props.get('id_b'), "props": rel_props}))
        return queries

    def upload_graph_from_json(self, file_path):
        file_manager = Manager_File()
        graph_data = file_manager.upload_file(file_path)
        if graph_data is None:
            return False
        self.delete_entire_graph()
        queries = Manager_Graph.convert_json_to_cypher(graph_data)
        for query, params in queries:
            self.run_query(query, params)
        log = Log("INFO", "graph", self.__class__.__name__, "upload_graph_from_json", "Uploaded graph from JSON data after clearing existing graph.")
        self.logger.add_log_obj(log)
        return True

    def search_nodes_by_property(self, label, property_name, value):
        query = f"MATCH (n:{label}) WHERE n.{property_name} = $value RETURN n"
        return self.run_query(query, {"value": value})

    def search_full_text(self, index_name, query_string):
        query = f"CALL db.index.fulltext.queryNodes('{index_name}', $query_string) YIELD node RETURN node"
        return self.run_query(query, {"query_string": query_string})

    def semantic_search(self, query_string):
        query = f"CALL db.index.fulltext.queryNodes('semanticIndex', $query_string) YIELD node RETURN node"
        return self.run_query(query, {"query_string": query_string})

    def graph_search(self, start_node_id, depth):
        query = f"MATCH (n) WHERE id(n) = $start_node_id WITH n CALL apoc.path.subgraphAll(n, {{maxLevel: $depth}}) YIELD nodes, relationships RETURN nodes, relationships"
        return self.run_query(query, {"start_node_id": start_node_id, "depth": depth})
