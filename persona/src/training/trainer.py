from meta.meta_singleton import Meta_Singleton
from log.logger import Logger, Log
from chat.manager_chat import Manager_Chat

class Trainer(metaclass=Meta_Singleton):
    def __init__(self):
        self.logger = Logger()


    def train():
        manager_chat = Manager_Chat()