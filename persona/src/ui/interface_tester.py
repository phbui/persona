from .interface import Interface

class Interface_Tester(Interface):
    """
    Derived class for the Tester tab.
    Inherits from BaseInterface and sets the title / description at init time.
    """
    def __init__(self):
        super().__init__(
            title="Tester Interface",
            description="This is where testing functionalities could be placed."
        )

    # Tester-specific overrides or additional methods
    def some_tester_specific_method(self):
        print("Tester-specific method called.")