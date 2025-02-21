from .interface import Interface

class Interface_Trainer(Interface):
    """
    Derived class for the Trainer tab.
    Inherits from BaseInterface and sets the title / description at init time.
    """
    def __init__(self):
        super().__init__(
            title="Trainer Interface",
            description="This is where training settings or controls could be placed."
        )

    # Override or extend base functionality here
    def some_trainer_specific_method(self):
        print("Trainer-specific method called.")