from .interface import Interface

class Interface_Trainer(Interface):
    def __init__(self):
        super().__init__(
            title="Trainer Interface",
            description="This is where training settings or controls could be placed."
        )

    def some_trainer_specific_method(self):
        print("Trainer-specific method called.")