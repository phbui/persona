from meta.meta_singleton import Meta_Singleton
from manager.ai.manager_llm import Manager_LLM

class Manager_Prompt(metaclass= Meta_Singleton):
    def __init__(self):
        self.manager_llm = Manager_LLM()

    def generate_response(self, rounds,  query_data, candidates):
        return ""

    def choose_response(self, response_1, response_2):
        return "", 0, 0