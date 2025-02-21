import sys
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QTabWidget,
)
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt
from .interface_trainer import Interface_Trainer
from .interface_logger import Interface_Logger
from .interface_tester import Interface_Tester
from ..log.logger import Logger

class Manager_Tab(QMainWindow):
    def __init__(self):
        super().__init__()
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("My Modern UI System (PyQt6)")
        self.resize(800, 600)

        # Create the tab widget and add the three interfaces
        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(Interface_Logger(), "Logger")
        self.tab_widget.addTab(Interface_Trainer(), "Trainer")
        self.tab_widget.addTab(Interface_Tester(), "Tester")

        # Set tab position (North is top)
        self.tab_widget.setTabPosition(QTabWidget.TabPosition.North)

        self.setCentralWidget(self.tab_widget)
        Logger().add_log("INFO", "ui", "Manager_Tab", "_setup_ui", "Setting up ui")

    def switch_tab(self, index: int):
        """
        Switch to the tab specified by index.
        0 = Logger, 1 = Trainer, 2 = Tester
        """
        if 0 <= index < self.tab_widget.count():
            self.tab_widget.setCurrentIndex(index)
            Logger().add_log("INFO", "ui", "Manager_Tab", "switch_tab", f"Switching to tab {index}")

    def closeEvent(self, event):
        """
        Override the default close event to fully shut down the application.
        """
        print("Manager_Tab close event triggered. Shutting down.")
        super().closeEvent(event)
        Logger().add_log("INFO", "ui", "Manager_Tab", "closeEvent", f"Shutting down")
        sys.exit(0)  # Force full shutdown
