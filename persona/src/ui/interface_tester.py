from .interface import Interface

class Interface_Tester(Interface):
    def __init__(self):
        super().__init__(
            title="Tester Interface",
            description="This is where testing functionalities could be placed."
        )

    def some_tester_specific_method(self):
        print("Tester-specific method called.")