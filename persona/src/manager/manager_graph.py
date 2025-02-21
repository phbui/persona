import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
from ..meta.meta_singleton import Meta_Singleton
from ..log.logger import Logger, Log

load_dotenv()
neo4js_uri = os.getenv('NEO4J_URI')
neo4js_user = os.getenv('NEO4J_USERNAME')
neo4js_password = os.getenv('NEO4J_PASSWORD')

class Manager_Graph(metaclass=Meta_Singleton):
    """
    A Singleton class to manage a connection to a Neo4j graph instance.
    Uses the Neo4j Python driver to connect via the Bolt protocol.
    """
    def __init__(self):
        # Construct the connection URI using your instance key.
        self.uri = neo4js_uri
        self.user = neo4js_user
        self.password = neo4js_password
        self.logger = Logger()
        
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            log = Log("INFO", "graph", self.__class__.__name__, "__init__", f"Connected to Neo4j instance at {self.uri}")
            self.logger.add_log_obj(log)
        except Exception as e:
            log = Log("ERROR", "graph", self.__class__.__name__, "__init__", f"Failed to connect to Neo4j at {self.uri}: {e}")
            self.logger.add_log_obj(log)
            raise

    def close(self):
        """Closes the connection to the Neo4j instance."""
        self.driver.close()
        log = Log("INFO", "graph", self.__class__.__name__, "close", "Connection closed.")
        self.logger.add_log_obj(log)

    def run_query(self, query, parameters=None):
        """
        Executes a Cypher query on the Neo4j instance.
        
        :param query: The Cypher query to run.
        :param parameters: Optional dictionary of parameters.
        :return: List of result records as dictionaries, or None if an error occurs.
        """
        if parameters is None:
            parameters = {}
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters)
                data = [record.data() for record in result]
                log = Log("INFO", "graph", self.__class__.__name__, "run_query", f"Executed query: {query} with parameters: {parameters}")
                self.logger.add_log_obj(log)
                return data
        except Exception as e:
            log = Log("ERROR", "graph", self.__class__.__name__, "run_query", f"Query failed: {query} with error: {e}")
            self.logger.add_log_obj(log)
            return None
