from .interface import Interface

class Interface_Logger(Interface):
    """
    Derived class for the Logger tab.
    Inherits from BaseInterface and sets the title / description at init time.
    """
    def __init__(self):
        super().__init__(
            title="Logger Interface",
            description="This is where logs or logging controls could be displayed."
        )

    # Logger-specific overrides or additional methods
    def some_logger_specific_method(self):
        print("Logger-specific method called.")