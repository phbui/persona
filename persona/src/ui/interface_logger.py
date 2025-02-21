from PyQt6.QtWidgets import QTabWidget, QVBoxLayout
from .interface import Interface
from .tab.tab_logger_browse import Tab_Logger_Browse

class Interface_Logger(Interface):
    """
    Derived class for the Logger tab.
    Inherits from Interface and sets the title / description at init time.
    This class acts as a container for sub-tabs, such as the "Browse Logs" tab.
    """
    def __init__(self):
        super().__init__(
            title="Logger Interface",
            description="This is where logs or logging controls could be displayed."
        )
        self.init_sub_tabs()

    def init_sub_tabs(self):
        """
        Initialize the sub-tabs within the Logger interface.
        """
        # Create a QTabWidget to hold sub-tabs
        self.tab_widget = QTabWidget()

        # Add the "Browse Logs" sub-tab
        self.tab_widget.addTab(Tab_Logger_Browse(), "Browse Logs")
        
        # You can add more sub-tabs here if needed.
        # e.g., self.tab_widget.addTab(AnotherTab(), "Another Tab")

        # Add the QTabWidget to the main layout of the interface.
        # Note: self.layout() is defined in the base Interface.
        main_layout = self.layout()
        main_layout.addWidget(self.tab_widget)
