import os
import re
import time
import numpy as np
from neo4j import GraphDatabase
from dotenv import load_dotenv
from meta.meta_singleton import Meta_Singleton
from log.logger import Logger, Log
from manager.manager_file import Manager_File
from manager.ai.manager_extraction import Manager_Extraction

load_dotenv()
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

def log_function(func):
    def wrapper(*args, **kwargs):
        self = args[0]
        self._log("DEBUG", func.__name__, f"ENTER: args={args[1:]}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            self._log("DEBUG", func.__name__, f"EXIT: returned {result}")
            return result
        except Exception as e:
            self._log("ERROR", func.__name__, f"Exception: {e}")
            raise
    return wrapper

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
            self._create_fulltext_index()
        except Exception as e:
            self._log("ERROR", "__init__", f"Failed to connect to Neo4j at {self.uri}: {e}")
            raise

    @log_function
    def _create_fulltext_index(self):
        query = (
            "CREATE FULLTEXT INDEX bm25Index IF NOT EXISTS "
            "FOR (n:Episode) ON EACH [n.content]; "
        )
        self.run_query(query)

    def _log(self, level, method_name, message):
        log_entry = Log(level, "graph", self.__class__.__name__, method_name, message)
        self.logger.add_log_obj(log_entry)

    @log_function
    def close(self):
        self.driver.close()
        self._log("INFO", "close", "Connection closed.")

    @log_function
    def run_query(self, query, parameters=None):
        parameters = parameters or {}
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters)
                data = [record.data() for record in result]
            self._log("INFO", "run_query", f"Executed query: {query} with parameters: {parameters} with result {data}")
            return data
        except Exception as e:
            self._log("ERROR", "run_query", f"Query failed: {query} with error: {e}")
            return None

    @log_function
    def _add_node(self, data_dict, node_label):
        query = (
            f"MERGE (n:{node_label} {{ content: $content }}) "
            "ON CREATE SET n.timestamp = $timestamp, n.embedding = $embedding, n.sentiment = $sentiment, n.emotion = $emotion"
        )
        params = {
            "content": data_dict.get("content"),
            "timestamp": data_dict.get("timestamp", time.time()),
            "embedding": data_dict.get("embedding"),
            "sentiment": data_dict.get("sentiment"),
            "emotion": data_dict.get("emotion")
        }
        self.run_query(query, params)
        self._log("INFO", "_add_node", f"Added {node_label} node with content: {data_dict.get('content')}")

    @log_function
    def _add_episode(self, episode_data):
        self._add_node(episode_data, node_label="Episode")

    @log_function
    def _add_entity(self, entity_data):
        self._add_node(entity_data, node_label="Entity")

    @log_function
    def _add_community(self, community_data):
        self._add_node(community_data, node_label="Community")

    @log_function
    def _add_memory_relationship(self, source_content, target_content, relationship_type="RELATED",
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

    @log_function
    def break_string(self, str):
        sentence_endings = re.compile(
            r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.{3}|[.!?])\s'
        )
        
        sentences = sentence_endings.split(str)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        
        return sentences


    @log_function
    def create_entire_graph(self, file_path):
        graph_data = Manager_File().upload_file(file_path)
        sentences = self.break_string(graph_data)
        for sentence in sentences:
            input_memory = {
                "content": sentence, 
                "timestamp": time.time()
                }

            self.process_new_memory(input_memory)


    @log_function
    def delete_entire_graph(self):
        self.run_query("MATCH (n) DETACH DELETE n")
        self._log("INFO", "delete_entire_graph", "Deleted the entire graph.")

    @log_function
    def download_entire_graph(self, directory_path, file_name="graph_download.json"):
        nodes = self.run_query("MATCH (n) RETURN labels(n) AS labels, properties(n) AS props") or []
        relationships = self.run_query(
            "MATCH ()-[r]->() RETURN type(r) AS type, id(startNode(r)) AS start_id, id(endNode(r)) AS end_id, properties(r) AS props"
        ) or []
        graph_data = {"nodes": nodes, "relationships": relationships}
        success = Manager_File().download_file(graph_data, directory_path, file_name)
        return success

    @log_function
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

    @log_function
    def upload_entire_graph(self, file_path):
        graph_data = Manager_File().upload_file(file_path)
        if graph_data is None:
            return False
        self.delete_entire_graph()
        for query, params in self._convert_json(graph_data):
            self.run_query(query, params)
        return True

    @log_function
    def _propagate_labels(self, entities, max_iterations=10, similarity_threshold=0.5):
        num_entities = len(entities)
        labels = list(range(num_entities))
        for _ in range(max_iterations):
            new_labels = labels.copy()
            for i in range(num_entities):
                neighbor_labels = []
                for j in range(num_entities):
                    if i != j:
                        similarity = self.manager_extraction.cosine_similarity(entities[i]["embedding"], entities[j]["embedding"])
                        if similarity >= similarity_threshold:
                            neighbor_labels.append(labels[j])
                if neighbor_labels:
                    new_labels[i] = max(set(neighbor_labels), key=neighbor_labels.count)
            if new_labels == labels:
                break
            labels = new_labels
        return labels

    @log_function
    def _build_semantic_subgraph(self, entities_list: list, similarity_threshold=0.5):
        for i in range(len(entities_list)):
            self._add_entity(entities_list[i])
            for j in range(i + 1, len(entities_list)):
                similarity = self.manager_extraction.cosine_similarity(entities_list[i]["embedding"], entities_list[j]["embedding"])
                if similarity >= similarity_threshold:
                    self._add_memory_relationship(
                        source_content=entities_list[i]["content"],
                        target_content=entities_list[j]["content"],
                        relationship_type="SEMANTICALLY_RELATED",
                        llm_edge=True,
                        relationship_weight=similarity,
                        source_label="Entity",
                        target_label="Entity"
                    )

    @log_function
    def _link_episode_to_entity(self, episode_content, entity_content):
        query = "MATCH (e:Episode {content: $episode_content}), (en:Entity {content: $entity_content}) MERGE (e)-[:EXTRACTS]->(en)"
        self.run_query(query, {"episode_content": episode_content, "entity_content": entity_content})
        self._log("INFO", "_link_episode_to_entity", f"Linked Episode '{episode_content}' to Entity '{entity_content}'")

    @log_function
    def process_new_memory(self, episode_data, context_window=5):
        episode_content = episode_data.get("content")
        episode_embedding = self.manager_extraction.extract_embedding(episode_content)
        episode_sentiment = self.manager_extraction.extract_sentiment(episode_content)
        episode_emotion = self.manager_extraction.extract_emotion(episode_content)

        episode_data["embedding"] = episode_embedding
        episode_data["sentiment"] = episode_sentiment
        episode_data["emotion"] = episode_emotion
        self._add_episode(episode_data)

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
            candidates = self.run_query(
                "MATCH (e:Entity) WHERE toLower(e.content) CONTAINS toLower($content) RETURN e.content AS content, e.embedding AS embedding",
                {"content": entity["content"]}
            )
            resolved_entity = self.manager_extraction.resolve_entity(entity, candidates)
            resolved_entities.append(resolved_entity)
            self._add_entity(resolved_entity)
            self._link_episode_to_entity(episode_content, resolved_entity["content"])
        self._log("INFO", "process_new_memory", f"Extracted and linked {len(resolved_entities)} entities.")
        self._update_entire_graph()

    @log_function
    def _update_semantic_subgraph(self):
        entities_data = self.run_query("MATCH (en:Entity) RETURN en.content AS content, en.embedding AS embedding, en.sentiment AS sentiment, en.emotion AS emotion")
        if not entities_data:
            return
        entities_list = [{"content": record["content"], "embedding": record["embedding"], "sentiment": record["sentiment"], "emotion": record["emotion"]} for record in entities_data]
        self._build_semantic_subgraph(entities_list, similarity_threshold=0.7)
        self._log("INFO", "_update_semantic_subgraph", "Semantic subgraph updated.")

    @log_function
    def _update_community_subgraph(self):
        entities_data = self.run_query("MATCH (en:Entity) RETURN en.content AS content, en.embedding AS embedding, en.sentiment AS sentiment, en.emotion AS emotion")
        if not entities_data:
            self._log("CRITICAL", "_update_community_subgraph", "No entities found for community update.")
            return
        entity_list = [{"content": record["content"], "embedding": record["embedding"], "sentiment": record["sentiment"], "emotion": record["emotion"]} for record in entities_data]
        labels = self._propagate_labels(entity_list, max_iterations=10)
        new_assignments = {}  # entity content -> community id
        clusters = {}
        for idx, label in enumerate(labels):
            community_id = f"community_{label}"
            new_assignments[entity_list[idx]["content"]] = community_id
            clusters.setdefault(community_id, []).append(entity_list[idx]["content"])

        current_results = self.run_query(
            "MATCH (en:Entity)-[r:BELONGS_TO]->(c:Community) RETURN en.content AS content, c.content AS community"
        )
        current_assignments = {record["content"]: record["community"] for record in current_results} if current_results else {}

        for entity, new_comm in new_assignments.items():
            current_comm = current_assignments.get(entity)
            if current_comm != new_comm:
                if current_comm:
                    self.run_query(
                        "MATCH (en:Entity {content: $entity})-[r:BELONGS_TO]->(c:Community {content: $current_comm}) DELETE r",
                        {"entity": entity, "current_comm": current_comm}
                    )
                self._add_community({"content": new_comm, "timestamp": time.time(), "embedding": None, "sentiment": None, "emotion": None})
                self.run_query(
                    "MATCH (en:Entity {content: $entity}), (c:Community {content: $new_comm}) MERGE (en)-[:BELONGS_TO]->(c)",
                    {"entity": entity, "new_comm": new_comm}
                )

        for entity in current_assignments.keys():
            if entity not in new_assignments:
                self.run_query(
                    "MATCH (en:Entity {content: $entity})-[r:BELONGS_TO]->(c:Community) DELETE r",
                    {"entity": entity}
                )

        self.run_query("MATCH (c:Community) WHERE NOT (c)<-[:BELONGS_TO]-(:Entity) DETACH DELETE c")
        self._log("INFO", "_update_community_subgraph", "Incremental community update completed.")

    @log_function
    def _update_entire_graph(self):
        self._update_community_subgraph()
        self._update_semantic_subgraph()
        self._log("INFO", "_update_entire_graph", "Updated entire graph with new connections.")

    @log_function
    def _search_index(self, index_name, query_text, result_limit=5):
        query = f"CALL db.index.fulltext.queryNodes('{index_name}', $query_text) YIELD node, score RETURN node, score LIMIT $result_limit"
        results = self.run_query(query, {"query_text": query_text, "result_limit": result_limit})
        candidates = {}
        for r in results or []:
            content = r["node"]["content"]
            timestamp = r["node"]["timestamp"]
            embedding = r["node"]["embedding"]
            sentiment = r["node"]["sentiment"]
            emotion = r["node"]["emotion"]
            candidates[content] = {
                "content": content, 
                "timestamp": timestamp, 
                "embedding": embedding, 
                "sentiment": sentiment, 
                "emotion": emotion, 
                f"{index_name}_score": r["score"]}
        return candidates

    @log_function
    def _semantic_search(self, query_text, result_limit=5):
        return self._search_index("semanticIndex", query_text, result_limit)

    @log_function
    def _bm25_search(self, query_text, result_limit=5):
        return self._search_index("bm25Index", query_text, result_limit)
    
    @log_function
    def _graph_search(self, start_node_id, depth):
        query = ("MATCH (n) WHERE id(n) = $start_node_id "
                 "WITH n CALL apoc.path.subgraphAll(n, {maxLevel: $depth}) YIELD nodes, relationships "
                 "RETURN nodes, relationships")
        return self.run_query(query, {"start_node_id": start_node_id, "depth": depth})

    @log_function
    def _graph_search_score(self, content, depth=1):
        id_res = self.run_query("MATCH (m) WHERE m.content = $content AND (m:Episode OR m:Entity OR m:Community) RETURN id(m) AS id LIMIT 1", {"content": content})
        if id_res and (nid := id_res[0].get("id")):
            gs = self._graph_search(nid, depth)
            return len(gs[0].get("nodes", [])) if gs and gs[0].get("nodes") else 0
        return 0

    @log_function
    def _bfs_search(self, seed_content, result_limit=5, depth=2):
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
        
    @log_function
    def merge_candidate(self, key, semantic_candidates, bm25_candidates, bfs_candidates):
        sem = semantic_candidates.get(key, {})
        bm25 = bm25_candidates.get(key, {})
        bfs = bfs_candidates.get(key, {})

        ts_sem = sem.get("timestamp", None)
        ts_bm25 = bm25.get("timestamp", None)

        if ts_sem is not None:
            merged_ts = ts_sem
        elif ts_bm25 is not None:
            merged_ts = ts_bm25
        else:
            merged_ts = 0
        timestamp = time.time() - merged_ts

        emb_sem = sem.get("embedding", None)
        emb_bm25 = bm25.get("embedding", None)

        if emb_sem is not None:
            merged_embedding = emb_sem
        elif emb_bm25 is not None:
            merged_embedding = emb_bm25
        else:
            merged_embedding = [0]

        sentiment_sem = sem.get("sentiment", None)
        sentiment_bm25 = bm25.get("sentiment", None)

        if sentiment_sem is not None:
            merged_sentiment = sentiment_sem
        elif sentiment_bm25 is not None:
            merged_sentiment = sentiment_bm25
        else:
            merged_sentiment = {}

        emotion_sem = sem.get("emotion", {})
        emotion_bm25 = bm25.get("emotion", {})

        merged_emotion = {}
        
        if emotion_sem:
            merged_emotion = emotion_sem
        elif emotion_bm25:
            merged_emotion = emotion_bm25

        # Scores
        semantic_score = sem.get("semanticIndex_score", 0)
        bm25_score = bm25.get("bm25Index_score", 0)
        graph_score = bfs.get("graph_score", 0)

        candidate = {
            "content": key,
            "timestamp": timestamp,
            "embedding": merged_embedding,
            "sentiment": merged_sentiment,
            "emotion": merged_emotion,
            "semantic_score": semantic_score,
            "bm25_score": bm25_score,
            "graph_score": graph_score,
        }
        return candidate

    @log_function
    def retrieve_candidates(self, query_text, result_limit=5):
        query_embedding = self.manager_extraction.extract_embedding(query_text)
        semantic_candidates = self._semantic_search(query_text, result_limit*2)
        bm25_candidates = self._bm25_search(query_text, result_limit*2)
        bfs_candidates = {}

        if semantic_candidates:
            seed_content = next(iter(semantic_candidates.values()))["content"]
            bfs_candidates = self._bfs_search(seed_content, result_limit*2, depth=2)
        all_keys = set(semantic_candidates.keys()) | set(bm25_candidates.keys()) | set(bfs_candidates.keys())
        all_keys = {key for key in all_keys if not key.startswith("community_")}
        all_candidates = {}

        for key in all_keys:
            candidate = self.merge_candidate(key, semantic_candidates, bm25_candidates, bfs_candidates)
            all_candidates[key] = candidate

        for candidate in all_candidates.values():
            if candidate["semantic_score"] == 0 or candidate["timestamp"] == 0:
                sem = self._search_index("semanticIndex", candidate["content"], result_limit=1)
                candidate["semantic_score"] = next(iter(sem.values()), {}).get("semanticIndex_score", candidate["semantic_score"])
            if candidate["bm25_score"] == 0:
                bm25 = self._search_index("bm25Index", candidate["content"], result_limit=1)
                candidate["bm25_score"] = next(iter(bm25.values()), {}).get("bm25Index_score", 0)
            if candidate["graph_score"] == 0:
                candidate["graph_score"] = self._graph_search_score(candidate["content"], depth=1)

            candidate["similarity"] = self.manager_extraction.cosine_similarity(query_embedding, candidate["embedding"])
            del candidate["embedding"]

        return list(all_candidates.values())