from .manager_policy import Manager_Policy

class Manager_RL():
    def __init__(self, policy):
        self.manager_policy = Manager_Policy(policy)
        pass