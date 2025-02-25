from .interface import Interface
from testing.tester import Tester

class Interface_Tester(Interface):
    def __init__(self):
        super().__init__(
            title="Tester Interface",
            description="This is where testing functionalities could be placed."
        )
        self.tester = Tester()

    def some_tester_specific_method(self):
        print("Tester-specific method called.")