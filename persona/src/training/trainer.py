from meta.meta_singleton import Meta_Singleton
from log.logger import Logger, Log
from manager.manager_graph import Manager_Graph
from chat.manager_chat import Manager_Chat
from agent.agent_rl import Agent_RL
from agent.agent_trainer import Agent_Trainer

class Trainer(metaclass=Meta_Singleton):
    def __init__(self):
        self.logger = Logger()
        self.manager_graph = Manager_Graph()
        self.policy = None

    def train(self, epochs, rounds, clip_range, learning_rate, discount_factor, gae_param):
        agent_rl = Agent_RL(name, self.policy)
        agent_trainer = Agent_Trainer(name)

        manager_chat = Manager_Chat(agent_trainer, agent_rl, epochs, rounds)

    def load_policy(self, file_path):
        print()

    def download_policy(self, file_path, file_name):
        print()

    def create_graph(self, file_path):
        self.manager_graph.create_entire_graph(file_path)

    def upload_graph(self, file_path):
        self.manager_graph.upload_entire_graph(file_path)

    def delete_graph(self):
        self.manager_graph.delete_entire_graph()

    def download_graph(self, file_path, file_name):
        self.manager_graph.download_entire_graph(file_path, file_name)