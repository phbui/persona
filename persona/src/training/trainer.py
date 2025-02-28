from meta.meta_singleton import Meta_Singleton
from log.logger import Logger, Log
from manager.manager_graph import Manager_Graph
from manager.manager_file import Manager_File
from chat.manager_chat import Manager_Chat
from agent.agent_rl import Agent_RL
from agent.agent_trainer import Agent_Trainer
import torch as th


class Trainer(metaclass=Meta_Singleton):
    def __init__(self):
        self.logger = Logger()
        self.manager_graph = Manager_Graph()
        self.agent_rl = None

    def train(self, epochs, rounds, clip_range, learning_rate, discount_factor, gae_param, trainer_agent_name, rl_agent_name):
        agent_trainer = Agent_Trainer(trainer_agent_name)
        agent_rl = Agent_RL(rl_agent_name,)

        manager_chat = Manager_Chat(agent_trainer, agent_rl, epochs, rounds)

    def _load_policy(self, file_path, policy):
        model = Manager_File().upload_file(file_path)
        state_dict = {k: th.tensor(v) for k, v in model.items()}
        policy.load_state_dict(state_dict)

    def _download_policy(self, file_path, file_name, policy):
        if self.agent_rl is None:
            Manager_File().download_file({}, file_path, file_name)
            return
        state_dict = policy.state_dict()
        serializable_dict = {k: v.cpu().tolist() for k, v in state_dict.items()}
        Manager_File().download_file(serializable_dict, file_path, file_name)

    def load_mem_policy(self, file_path):
        self._load_policy(file_path, self.model.policy_mem)

    def download_mem_policy(self, file_path, file_name):
        self._download_policy(file_path, file_name, self.agent_rl.policy_mem)

    def load_emo_policy(self, file_path):
        self._load_policy(file_path, self.model.policy_emo)

    def download_emo_policy(self, file_path, file_name):
        self._download_policy(file_path, file_name, self.agent_rl.policy_emo)

    def create_graph(self, file_path):
        self.manager_graph.create_entire_graph(file_path)

    def upload_graph(self, file_path):
        self.manager_graph.upload_entire_graph(file_path)

    def delete_graph(self):
        self.manager_graph.delete_entire_graph()

    def download_graph(self, file_path, file_name):
        self.manager_graph.download_entire_graph(file_path, file_name)