from PyQt6.QtWidgets import QTabWidget
from .interface import Interface
from .tab.tab_logger_browse import Tab_Logger_Browse

class Interface_Logger(Interface):
    def __init__(self):
        super().__init__(
            title="Logger Interface",
            description="This is where logs or logging controls could be displayed."
        )
        self.init_sub_tabs()

    def init_sub_tabs(self):
        self.tab_widget = QTabWidget()

        self.tab_widget.addTab(Tab_Logger_Browse(), "Browse Logs")

        main_layout = self.layout()
        main_layout.addWidget(self.tab_widget)
