from meta.meta_singleton import Meta_Singleton
from log.logger import Logger, Log
from manager.manager_graph import Manager_Graph
from chat.manager_chat import Manager_Chat

class Trainer(metaclass=Meta_Singleton):
    def __init__(self):
        self.logger = Logger()
        self.manager_graph = Manager_Graph()

    def train(self):
        manager_chat = Manager_Chat()


    def create_graph(self, txt):
        self.manager_graph.create_entire_graph(txt)

    def upload_graph(self, dir):
        self.manager_graph.upload_entire_graph(dir)

    def delete_graph(self):
        self.manager_graph.delete_entire_graph()

    def download_graph(self, dir):
        self.manager_graph.download_entire_graph(dir)