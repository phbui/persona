from meta.meta_singleton import Meta_Singleton
from log.logger import Logger, Log

class Tester(metaclass=Meta_Singleton):
    def __init__(self):
        self.logger = Logger()