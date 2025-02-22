from typing import Dict, Any, List
import json
from meta.meta_singleton import Meta_Singleton
from log.logger import Logger, Log
from manager.ai.manager_llm import Manager_LLM

class Manager_Extraction(metaclass=Meta_Singleton):
    def __init__(self):
        self.manager_llm = Manager_LLM()
        self.logger = Logger()
        
    def _log(self, level, method, message):
        log = Log(level, "extraction", self.__class__.__name__, method, message)
        self.logger.add_log_obj(log)

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        prompt = "Extract entities from the following text. Return a JSON list of objects with keys 'content' and 'embedding'. Text: " + text
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
        prompt = "Extract facts from the following text. Return a JSON list of objects with keys 'fact' and 'embedding'. Text: " + text
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
