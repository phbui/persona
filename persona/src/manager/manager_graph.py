import os
import re
import time
import random
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
            "FOR (n:Memory) ON EACH [n.content]; "
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
    def _add_memory(self, memory_data):
        self._add_node(memory_data, node_label="Memory")

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
    def break_string(self, text):
        # Regular expression pattern to detect sentence endings while preserving quoted text
        sentence_endings = re.compile(
            r'([.!?])\s+(?=(?:[^"]*"[^"]*")*[^"]*$)'
        )

        # Split text while keeping delimiters (punctuation) attached to sentences
        sentences = sentence_endings.split(text)

        # Recombine sentences while keeping punctuation attached
        cleaned_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            cleaned_sentences.append(sentences[i] + sentences[i + 1])

        # Ensure no empty strings remain
        cleaned_sentences = [s.strip() for s in cleaned_sentences if s.strip()]
        
        return cleaned_sentences

    
    @log_function
    def text_to_memories(self, text, context_window = 5):
        sentences = self.break_string(text)
        for sentence in sentences:
            input_memory = {
                "content": sentence, 
                "timestamp": time.time()
                }

            self.process_new_memory(input_memory, context_window)

    @log_function
    def create_entire_graph(self, file_path, context_window = 5):
        graph_data = Manager_File().upload_file(file_path)
        self.text_to_memories(graph_data, context_window)


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
    def _link_memory_to_entity(self, memory_content, entity_content):
        query = "MATCH (e:Memory {content: $memory_content}), (en:Entity {content: $entity_content}) MERGE (e)-[:EXTRACTS]->(en)"
        self.run_query(query, {"memory_content": memory_content, "entity_content": entity_content})
        self._log("INFO", "_link_memory_to_entity", f"Linked Memory '{memory_content}' to Entity '{entity_content}'")

    @log_function
    def _process_entities(self, memory_content):
        extracted_entities = []
        try:
            extracted_entities = self.manager_extraction.extract_entities(memory_content)
        except Exception as e:
            self._log("ERROR", "_process_new_entities_from_memory", f"Entity extraction failed: {e}")

        resolved_entities = []
        for entity in extracted_entities:
            candidates = self.run_query(
                "MATCH (e:Entity) WHERE toLower(e.content) CONTAINS toLower($content) RETURN e.content AS content, e.embedding AS embedding",
                {"content": entity["content"]}
            )
            resolved_entity = self.manager_extraction.resolve_entity(entity, candidates)
            resolved_entities.append(resolved_entity)
            self._add_entity(resolved_entity)
            self._link_memory_to_entity(memory_content, resolved_entity["content"])

        return resolved_entities
    
    @log_function
    def _process_memory(self, memory_content, context_window=5):
        previous_memories = self.run_query(
            "MATCH (e:Memory) ORDER BY e.timestamp DESC RETURN e.content AS content, e.embedding AS embedding, e.sentiment AS sentiment, e.emotion AS emotion LIMIT $limit",
            {"limit": context_window}
        )
        context_text = memory_content

        if previous_memories:
            context_text += " " + " ".join([record["content"] for record in previous_memories])

        memory = self.manager_extraction.extract_memory(memory_content)

        resolved_memory = self.manager_extraction.resolve_memory(memory, previous_memories)

        if resolved_memory is not None:
            memory_data = {'timestamp': time.time()}
            memory_data["content"] = resolved_memory.get("content")
            memory_data["embedding"] = resolved_memory.get("embedding")
            memory_data["sentiment"] = resolved_memory.get("sentiment")
            memory_data["emotion"] = resolved_memory.get("emotion")

            self._add_memory(memory_data)

        return resolved_memory

    @log_function
    def process_new_memory(self, memory_data, context_window=5):
        memory_content = memory_data.get("content")
        self._log("INFO", "process_new_memory", f"Processing new memory: {memory_content}")
        memory = self._process_memory(memory_content, context_window)

        if memory is not None:
            self._process_entities(memory.get("content"))
            self._update_entire_graph()

    @log_function
    def _update_semantic_subgraph(self):
        entities_data = self.run_query("MATCH (en:Entity) RETURN en.content AS content, en.embedding AS embedding, en.sentiment AS sentiment, en.emotion AS emotion")
        if not entities_data:
            return
        entities_list = [{"content": record["content"], "embedding": record["embedding"], "sentiment": record["sentiment"], "emotion": record["emotion"]} for record in entities_data]
        self._build_semantic_subgraph(entities_list)
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
    def _format_candidate(self, node, score, score_label):
        return {
            "content": node["content"],
            "timestamp": node["timestamp"],
            "embedding": node["embedding"],
            "sentiment": node["sentiment"],
            "emotion": node["emotion"],
            f"{score_label}": score
        }
    
    @log_function
    def _semantic_search(self, query_embedding, result_limit=5):
        query = (
            "MATCH (n:Memory) "
            "WHERE n.embedding IS NOT NULL "
            "WITH n, gds.similarity.cosine(n.embedding, $query_embedding) AS similarity "
            "RETURN n, similarity AS score "
            "ORDER BY score DESC "
            "LIMIT $result_limit"
        )
        results = self.run_query(query, {"query_embedding": query_embedding, "result_limit": result_limit})
        
        candidates = {}
        for r in results or []:
            node = r["n"]
            candidate = self._format_candidate(node, r["score"], "semantic_score")
            candidates[candidate["content"]] = candidate
        return candidates


    @log_function
    def _bm25_search(self, query_text, result_limit=5):
        query = (
            "CALL db.index.fulltext.queryNodes('bm25Index', $query_text) "
            "YIELD node, score "
            "WHERE node:Memory "
            "RETURN node, score LIMIT $result_limit"
        )
        results = self.run_query(query, {"query_text": query_text, "result_limit": result_limit})
        
        candidates = {}
        for r in results or []:
            node = r["node"]
            candidate = self._format_candidate(node, r["score"], "bm25_score")
            candidates[candidate["content"]] = candidate
        return candidates

    @log_function
    def retrieve_candidates(self, query_text, result_limit=10):
        query_embedding = self.manager_extraction.extract_embedding(query_text)
        
        bm25_candidates = self._bm25_search(query_text, result_limit)
        sem_candidates = self._semantic_search(query_embedding, result_limit)
        
        combined_candidates = {}
        
        for content, candidate in bm25_candidates.items():
            if content in sem_candidates:
                candidate["semantic_score"] = sem_candidates[content]["semantic_score"]
            else:
                candidate["semantic_score"] = self.manager_extraction.cosine_similarity(query_embedding, candidate["embedding"])
            candidate.pop("embedding", None)
            combined_candidates[content] = candidate
            
        for content, candidate in sem_candidates.items():
            if content not in combined_candidates:
                candidate["bm25_score"] = 0
                candidate.pop("embedding", None)
                combined_candidates[content] = candidate
                
        return random.sample(list(combined_candidates.values()), min(result_limit, len(combined_candidates)))
