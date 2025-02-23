import os
import time
import numpy as np
from neo4j import GraphDatabase
from dotenv import load_dotenv
from meta.meta_singleton import Meta_Singleton
from log.logger import Logger, Log
from manager.manager_file import Manager_File
from manager.manager_extraction import Manager_Extraction

load_dotenv()
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

class Manager_Graph(metaclass=Meta_Singleton):
    def __init__(self):
        self.uri = NEO4J_URI
        self.user = NEO4J_USER
        self.password = NEO4J_PASSWORD
        self.logger = Logger()
        self.manager_extraction = Manager_Extraction()
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            self._log("INFO", "__init__", f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            self._log("ERROR", "__init__", f"Failed to connect to Neo4j at {self.uri}: {e}")
            raise

    def _log(self, level, method_name, message):
        log_entry = Log(level, "graph", self.__class__.__name__, method_name, message)
        self.logger.add_log_obj(log_entry)

    def close(self):
        self.driver.close()
        self._log("INFO", "close", "Connection closed.")

    def run_query(self, query, parameters=None):
        parameters = parameters or {}
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters)
                data = [record.data() for record in result]
            self._log("INFO", "run_query", f"Executed query: {query} with parameters: {parameters}")
            return data
        except Exception as e:
            self._log("ERROR", "run_query", f"Query failed: {query} with error: {e}")
            return None

    def add_node(self, data_dict, node_label):
        query = (
            f"MERGE (n:{node_label} {{ content: $content }}) "
            "ON CREATE SET n.timestamp = $timestamp, n.embedding = $embedding"
        )
        params = {
            "content": data_dict.get("content"),
            "timestamp": data_dict.get("timestamp", time.time()),
            "embedding": data_dict.get("embedding")
        }
        self.run_query(query, params)
        self._log("INFO", "add_node", f"Added {node_label} node with content: {data_dict.get('content')}")

    def add_episode(self, episode_data):
        self.add_node(episode_data, node_label="Episode")

    def add_entity(self, entity_data):
        self.add_node(entity_data, node_label="Entity")
        self.update_community_for_entity(entity_data["content"])

    def add_community(self, community_data):
        self.add_node(community_data, node_label="Community")

    def update_community_for_entity(self, entity_content):
        query = (
            "MATCH (e:Entity {content: $content})-->(n) "
            "WHERE exists(n.community_id) "
            "RETURN n.community_id AS cid, count(*) AS freq "
            "ORDER BY freq DESC LIMIT 1"
        )
        result = self.run_query(query, {"content": entity_content})
        if result and len(result) > 0:
            community_id = result[0]["cid"]
            self._log("INFO", "update_community_for_entity", f"Found existing community {community_id} for entity {entity_content}")
        else:
            community_id = f"community_{int(time.time())}"
            self.add_community({"content": community_id, "timestamp": time.time(), "embedding": None})
            self._log("INFO", "update_community_for_entity", f"Created new community {community_id} for entity {entity_content}")
        update_query = (
            "MATCH (e:Entity {content: $content}) "
            "SET e.community_id = $community_id"
        )
        self.run_query(update_query, {"content": entity_content, "community_id": community_id})
        self._log("INFO", "update_community_for_entity", f"Assigned community {community_id} to entity {entity_content}")

    def add_memory_relationship(self, source_content, target_content, relationship_type="RELATED",
                                  llm_edge=False, relationship_weight=1.0, source_label="Memory", target_label="Memory"):
        query = (
            f"MATCH (m1:{source_label} {{content: $source_content}}), (m2:{target_label} {{content: $target_content}}) "
            f"MERGE (m1)-[r:{relationship_type}]->(m2) "
            "ON CREATE SET r.llm_edge = $llm_edge, r.weight = $relationship_weight"
        )
        params = {
            "source_content": source_content,
            "target_content": target_content,
            "llm_edge": llm_edge,
            "relationship_weight": relationship_weight
        }
        self.run_query(query, params)

    def retrieve_memory(self, query_text, limit=5):
        query = ("CALL db.index.fulltext.queryNodes('memoryIndex', $query_text) "
                 "YIELD node, score RETURN node, score LIMIT $limit")
        results = self.run_query(query, {"query_text": query_text, "limit": limit})
        count = len(results) if results else 0
        self._log("INFO", "retrieve_memory", f"Retrieved {count} nodes for query: {query_text}")
        return results

    def graph_search(self, start_node_id, depth):
        query = ("MATCH (n) WHERE id(n) = $start_node_id "
                 "WITH n CALL apoc.path.subgraphAll(n, {maxLevel: $depth}) YIELD nodes, relationships "
                 "RETURN nodes, relationships")
        return self.run_query(query, {"start_node_id": start_node_id, "depth": depth})

    def delete_entire_graph(self):
        self.run_query("MATCH (n) DETACH DELETE n")
        self._log("INFO", "delete_entire_graph", "Deleted the entire graph.")

    def download_entire_graph(self, directory_path, file_name="graph_download.json"):
        nodes = self.run_query("MATCH (n) RETURN labels(n) AS labels, properties(n) AS props") or []
        relationships = self.run_query(
            "MATCH ()-[r]->() RETURN type(r) AS type, id(startNode(r)) AS start_id, id(endNode(r)) AS end_id, properties(r) AS props"
        ) or []
        graph_data = {"nodes": nodes, "relationships": relationships}
        success = Manager_File().download_file(graph_data, directory_path, file_name)
        return success

    def _convert_json(self, graph_data):
        queries = []
        for node in graph_data.get("nodes", []):
            label = node.get("labels", ["Memory"])[0]
            queries.append((f"CREATE (n:{label} $props)", {"props": node.get("props", {})}))
        for rel in graph_data.get("relationships", []):
            if None not in (rel.get("type"), rel.get("start_id"), rel.get("end_id")):
                q = ("MATCH (a), (b) WHERE id(a) = $start_id AND id(b) = $end_id "
                     f"CREATE (a)-[r:{rel.get('type')} $props]->(b)")
                queries.append((q, {"start_id": rel["start_id"], "end_id": rel["end_id"], "props": rel.get("props", {})}))
        return queries

    def upload_entire_graph(self, file_path):
        graph_data = Manager_File().upload_file(file_path)
        if graph_data is None:
            return False
        self.delete_entire_graph()
        for query, params in self._convert_json(graph_data):
            self.run_query(query, params)
        return True

    def propagate_labels(self, entities, max_iterations=10):
        num_entities = len(entities)
        labels = list(range(num_entities))
        for _ in range(max_iterations):
            new_labels = labels.copy()
            for i in range(num_entities):
                neighbor_labels = []
                for j in range(num_entities):
                    if i != j:
                        similarity = self.manager_extraction.cosine_similarity(entities[i]["embedding"], entities[j]["embedding"])
                        if similarity >= 0.7:
                            neighbor_labels.append(labels[j])
                if neighbor_labels:
                    new_labels[i] = max(set(neighbor_labels), key=neighbor_labels.count)
            if new_labels == labels:
                break
            labels = new_labels
        return labels

    def summarize_community(self, community_entity_contents):
        aggregated_text = " ".join(community_entity_contents)
        prompt = "Summarize the following entities to capture their high-level context:\n" + aggregated_text
        summary = self.manager_extraction.manager_llm.generate_response(prompt, max_new_tokens=150, temperature=0.3)
        return summary.strip()

    def dynamic_community_update(self):
        entities_data = self.run_query("MATCH (en:Entity) RETURN en.content AS content, en.embedding AS embedding")
        if not entities_data:
            self._log("INFO", "dynamic_community_update", "No entities found for community update.")
            return
        entity_list = [{"content": record["content"], "embedding": record["embedding"]} for record in entities_data]
        labels = self.propagate_labels(entity_list, max_iterations=10)
        clusters = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(label, []).append(entity_list[idx]["content"])
        for cluster_id, contents in clusters.items():
            community_id = f"community_{cluster_id}"
            summary = self.summarize_community(contents)
            self.add_community({"content": community_id, "timestamp": time.time(), "embedding": None, "summary": summary})
            self.run_query("MATCH (c:Community {content: $community_id}), (en:Entity) WHERE en.content IN $contents MERGE (en)-[:BELONGS_TO]->(c)", {"community_id": community_id, "contents": contents})
        self._log("INFO", "dynamic_community_update", "Community update completed with iterative summarization.")

    def build_community_subgraph(self, num_clusters=5):
        nodes = self.run_query("MATCH (en:Entity) RETURN en.content AS content, en.embedding AS embedding")
        if not nodes:
            return
        entities = [{"content": record["content"], "embedding": record["embedding"]} for record in nodes]
        clusters = self.cluster_memory_nodes(entities, threshold=0.7, num_iterations=10)
        for cluster_id, contents in clusters.items():
            community_id = f"community_{cluster_id}"
            self.add_community({"content": community_id, "timestamp": time.time(), "embedding": None})
            for content in contents:
                self.run_query("MATCH (c:Community {content: $community_id}), (en:Entity {content: $entity_content}) MERGE (en)-[:BELONGS_TO]->(c)", {"community_id": community_id, "entity_content": content})

    def build_semantic_subgraph(self, entities_list: list, similarity_threshold=0.7):
        for i in range(len(entities_list)):
            self.add_entity(entities_list[i])
            for j in range(i + 1, len(entities_list)):
                similarity = self.manager_extraction.cosine_similarity(entities_list[i]["embedding"], entities_list[j]["embedding"])
                if similarity >= similarity_threshold:
                    self.add_memory_relationship(
                        source_content=entities_list[i]["content"],
                        target_content=entities_list[j]["content"],
                        relationship_type="SEMANTICALLY_RELATED",
                        llm_edge=True,
                        relationship_weight=similarity,
                        source_label="Entity",
                        target_label="Entity"
                    )

    def link_episode_to_entity(self, episode_content, entity_content):
        query = "MATCH (e:Episode {content: $episode_content}), (en:Entity {content: $entity_content}) MERGE (e)-[:EXTRACTS]->(en)"
        self.run_query(query, {"episode_content": episode_content, "entity_content": entity_content})
        self._log("INFO", "link_episode_to_entity", f"Linked Episode '{episode_content}' to Entity '{entity_content}'")

    def process_new_memory(self, episode_data, context_window=4):
        self.add_episode(episode_data)
        episode_content = episode_data.get("content")
        self._log("INFO", "process_new_memory", f"Processing new memory: {episode_content}")
        previous_episodes = self.run_query(
            "MATCH (e:Episode) WHERE e.timestamp < $current_timestamp ORDER BY e.timestamp DESC LIMIT $limit",
            {"current_timestamp": time.time(), "limit": context_window}
        )
        context_text = episode_content
        if previous_episodes:
            context_text += " " + " ".join([record["n"]["content"] for record in previous_episodes if "n" in record])
        try:
            extracted_entities = self.manager_extraction.extract_entities(context_text)
        except Exception as e:
            self._log("ERROR", "process_new_memory", f"Entity extraction failed: {e}")
            extracted_entities = []
        resolved_entities = []
        for entity in extracted_entities:
            resolved_entity = self.manager_extraction.resolve_entity(entity)
            resolved_entities.append(resolved_entity)
            self.add_entity(resolved_entity)
            self.link_episode_to_entity(episode_content, resolved_entity["content"])
        self._log("INFO", "process_new_memory", f"Extracted and linked {len(resolved_entities)} entities.")
        self.update_entire_graph()

    def update_semantic_subgraph(self):
        entities_data = self.run_query("MATCH (en:Entity) RETURN en.content AS content, en.embedding AS embedding")
        if not entities_data:
            return
        entities_list = [{"content": record["content"], "embedding": record["embedding"]} for record in entities_data]
        self.build_semantic_subgraph(entities_list, similarity_threshold=0.7)
        self._log("INFO", "update_semantic_subgraph", "Semantic subgraph updated.")

    def update_entire_graph(self):
        self.dynamic_community_update()
        self.update_semantic_subgraph()
        self._log("INFO", "update_entire_graph", "Updated entire graph with new connections.")

    def build_episode_subgraph(self, conversation_rounds: list):
        episode_id = f"episode_{int(time.time())}"
        self.run_query("MERGE (e:Episode {episode_id: $episode_id})", {"episode_id": episode_id})
        previous_content = None
        for round_data in conversation_rounds:
            self.add_episode(round_data)
            self.run_query(
                "MATCH (e:Episode {episode_id: $episode_id}), (ep:Episode {content: $content}) MERGE (ep)-[:PART_OF]->(e)",
                {"episode_id": episode_id, "content": round_data["content"]}
            )
            if previous_content:
                self.add_memory_relationship(
                    source_content=previous_content,
                    target_content=round_data["content"],
                    relationship_type="NEXT",
                    llm_edge=False,
                    relationship_weight=1.0,
                    source_label="Episode",
                    target_label="Episode"
                )
            previous_content = round_data["content"]

    def _search_index(self, index_name, query_text, result_limit=5):
        query = f"CALL db.index.fulltext.queryNodes('{index_name}', $query_text) YIELD node, score RETURN node, score LIMIT $result_limit"
        results = self.run_query(query, {"query_text": query_text, "result_limit": result_limit})
        candidates = {}
        for r in results or []:
            content = r["node"]["content"]
            candidates[content] = {"content": content, f"{index_name}_score": r["score"]}
        return candidates

    def semantic_search(self, query_text, result_limit=5):
        return self._search_index("semanticIndex", query_text, result_limit)

    def bm25_search(self, query_text, result_limit=5):
        return self._search_index("bm25Index", query_text, result_limit)

    def bfs_search(self, seed_content, result_limit=5, depth=2):
        id_result = self.run_query("MATCH (m) WHERE m.content = $content RETURN id(m) AS id LIMIT 1", {"content": seed_content})
        if id_result and (node_id := id_result[0].get("id")):
            query = (
                "MATCH (start) WHERE id(start) = $node_id "
                "CALL apoc.path.subgraphNodes(start, {maxLevel: $depth}) YIELD node "
                "RETURN node LIMIT $result_limit"
            )
            results = self.run_query(query, {"node_id": node_id, "depth": depth, "result_limit": result_limit})
            candidates = {}
            for r in results or []:
                content = r["node"]["content"]
                candidates[content] = {"content": content, "graph_score": self._graph_search_score(content, depth=1)}
            return candidates
        return {}

    def retrieve_candidates(self, query_text, result_limit=5):
        semantic_candidates = self.semantic_search(query_text, result_limit)
        bm25_candidates = self.bm25_search(query_text, result_limit)
        bfs_candidates = {}
        if semantic_candidates:
            seed_content = next(iter(semantic_candidates.values()))["content"]
            bfs_candidates = self.bfs_search(seed_content, result_limit, depth=2)
        all_keys = set(semantic_candidates.keys()) | set(bm25_candidates.keys()) | set(bfs_candidates.keys())
        all_candidates = {
            key: {
                "content": key,
                "semantic_score": semantic_candidates.get(key, {}).get("semanticIndex_score", 0),
                "bm25_score": bm25_candidates.get(key, {}).get("bm25Index_score", 0),
                "graph_score": bfs_candidates.get(key, {}).get("graph_score", 0)
            } for key in all_keys
        }
        for candidate in all_candidates.values():
            if candidate["semantic_score"] == 0:
                sem = self._search_index("semanticIndex", candidate["content"], result_limit=1)
                candidate["semantic_score"] = next(iter(sem.values()), {}).get("semanticIndex_score", 0)
            if candidate["bm25_score"] == 0:
                bm25 = self._search_index("bm25Index", candidate["content"], result_limit=1)
                candidate["bm25_score"] = next(iter(bm25.values()), {}).get("bm25Index_score", 0)
            if candidate["graph_score"] == 0:
                candidate["graph_score"] = self._graph_search_score(candidate["content"], depth=1)
        return list(all_candidates.values())

    def cluster_memory_nodes(self, memory_nodes, similarity_threshold=0.7, max_iterations=10):
        num_nodes = len(memory_nodes)
        labels = list(range(num_nodes))
        for _ in range(max_iterations):
            new_labels = labels.copy()
            for i in range(num_nodes):
                neighbor_counts = {}
                for j in range(num_nodes):
                    if i == j:
                        continue
                    similarity = self.manager_extraction.cosine_similarity(
                        memory_nodes[i]["embedding"], memory_nodes[j]["embedding"]
                    )
                    if similarity >= similarity_threshold:
                        neighbor_counts[labels[j]] = neighbor_counts.get(labels[j], 0) + 1
                if neighbor_counts:
                    new_labels[i] = max(neighbor_counts, key=neighbor_counts.get)
            if new_labels == labels:
                break
            labels = new_labels
        clusters = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(label, []).append(memory_nodes[idx]["content"])
        return clusters
