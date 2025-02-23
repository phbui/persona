import json
import numpy as np
from typing import Dict, Any, List
from meta.meta_singleton import Meta_Singleton
from log.logger import Logger, Log
from manager.ai.manager_llm import Manager_LLM

class Manager_Extraction(metaclass=Meta_Singleton):
    def __init__(self):
        self.manager_llm = Manager_LLM("google/flan-t5-base")
        self.logger = Logger()

    def _log(self, level, method, message):
        log = Log(level, "extraction", self.__class__.__name__, method, message)
        self.logger.add_log_obj(log)

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        prompt = (
            "Extract the named entities from the following text. "
            "Return ONLY a JSON array where each object has two keys: 'content' and 'embedding'. "
            "Set the 'embedding' value to null (it will be populated later). "
            "For example, if the text is: 'Alice visited Paris.' then the correct output would be: "
            "[{\"content\": \"Alice\", \"embedding\": null}, {\"content\": \"Paris\", \"embedding\": null}]. "
            "Text: " + text
        )
        self._log("INFO", "extract_entities", f"Sending prompt: {prompt}")
        response = self.manager_llm.generate_response(prompt, max_new_tokens=256, temperature=0.2)
        self._log("INFO", "extract_entities", f"Received response: {response}")
        try:
            entities = json.loads(response)
            if isinstance(entities, list):
                self._log("INFO", "extract_entities", f"Extracted {len(entities)} entities.")
                return entities
            self._log("WARNING", "extract_entities", "Response is not a list.")
            return []
        except Exception as e:
            self._log("ERROR", "extract_entities", f"Failed to parse JSON: {e}")
            return []
        
    def extract_facts(self, text: str) -> List[Dict[str, Any]]:
        prompt = (
            "Extract the factual statements from the following text. "
            "Return ONLY a JSON array where each object has two keys: 'fact' and 'embedding'. "
            "Set the 'embedding' value to null (it will be populated later). "
            "For example, if the text is: 'Alice visited Paris and Bob is a teacher.' then the correct output would be: "
            "[{\"fact\": \"Alice visited Paris\", \"embedding\": null}, {\"fact\": \"Bob is a teacher\", \"embedding\": null}]. "
            "Text: " + text
        )
        self._log("INFO", "extract_facts", f"Sending prompt: {prompt}")
        response = self.manager_llm.generate_response(prompt, max_new_tokens=256, temperature=0.2)
        self._log("INFO", "extract_facts", f"Received response: {response}")
        try:
            facts = json.loads(response)
            if isinstance(facts, list):
                self._log("INFO", "extract_facts", f"Extracted {len(facts)} facts.")
                return facts
            self._log("WARNING", "extract_facts", "Response is not a list.")
            return []
        except Exception as e:
            self._log("ERROR", "extract_facts", f"Failed to parse JSON: {e}")
            return []

    def resolve_entity(self, entity):
        candidates = self.run_query(
            "MATCH (e:Entity) WHERE toLower(e.content) CONTAINS toLower($content) RETURN e.content AS content, e.embedding AS embedding",
            {"content": entity["content"]}
        )
        if not candidates:
            return entity
        best_candidate = None
        best_sim = 0
        for candidate in candidates:
            sim = self.cosine_similarity(entity["embedding"], candidate["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_candidate = candidate
        if best_candidate and best_sim > 0.8:
            prompt = f"Are the following two entities redundant? Entity A: {entity['content']}. Entity B: {best_candidate['content']}. Answer yes or no."
            answer = self.manager_llm.generate_response(prompt, max_new_tokens=16, temperature=0.2)
            if "yes" in answer.lower():
                return best_candidate
        return entity
    
    def cosine_similarity(self, vec1, vec2):
        vec1, vec2 = np.array(vec1), np.array(vec2)
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        return float(np.dot(vec1, vec2) / (norm1 * norm2)) if norm1 and norm2 else 0.0

