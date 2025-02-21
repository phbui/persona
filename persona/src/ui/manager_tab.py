import sys
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QTabWidget,
)
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt
from ..meta.meta_singleton import Meta_Singleton
from .interface_trainer import Interface_Trainer
from .interface_logger import Interface_Logger
from .interface_tester import Interface_Tester

class Manager_Tab(QMainWindow):
    def __init__(self):
        super().__init__()
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("My Modern UI System (PyQt6)")
        self.resize(800, 600)

        # Create the tab widget and add the three interfaces
        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(Interface_Trainer(), "Trainer")
        self.tab_widget.addTab(Interface_Logger(), "Logger")
        self.tab_widget.addTab(Interface_Tester(), "Tester")

        # Set tab position (North is top)
        self.tab_widget.setTabPosition(QTabWidget.TabPosition.North)

        self.setCentralWidget(self.tab_widget)
        set_dark_fusion_style()

    def switch_tab(self, index: int):
        """
        Switch to the tab specified by index.
        0 = Trainer, 1 = Logger, 2 = Tester
        """
        if 0 <= index < self.tab_widget.count():
            self.tab_widget.setCurrentIndex(index)

    def closeEvent(self, event):
        """
        Override the default close event to fully shut down the application.
        """
        print("Manager_Tab close event triggered. Shutting down.")
        super().closeEvent(event)
        sys.exit(0)  # Force full shutdown

def set_dark_fusion_style():
    """
    Apply a dark Fusion style to give the application a sleek/modern look in PyQt6.
    """
    QApplication.setStyle("Fusion")

    # Create and configure a dark palette
    dark_palette = QPalette()
    dark_color = QColor(53, 53, 53)
    highlight_color = QColor(142, 45, 197).lighter()

    dark_palette.setColor(QPalette.ColorGroup.Active,   QPalette.ColorRole.Window,        dark_color)
    dark_palette.setColor(QPalette.ColorGroup.Inactive, QPalette.ColorRole.Window,        dark_color)
    dark_palette.setColor(QPalette.ColorGroup.Active,   QPalette.ColorRole.Base,          QColor(15, 15, 15))
    dark_palette.setColor(QPalette.ColorGroup.Inactive, QPalette.ColorRole.Base,          QColor(15, 15, 15))

    dark_palette.setColor(QPalette.ColorGroup.Active,   QPalette.ColorRole.WindowText,   Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorGroup.Active,   QPalette.ColorRole.Text,         Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorGroup.Inactive, QPalette.ColorRole.WindowText,   Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorGroup.Inactive, QPalette.ColorRole.Text,         Qt.GlobalColor.white)

    dark_palette.setColor(QPalette.ColorGroup.Active,   QPalette.ColorRole.Button,       dark_color)
    dark_palette.setColor(QPalette.ColorGroup.Inactive, QPalette.ColorRole.Button,       dark_color)
    dark_palette.setColor(QPalette.ColorGroup.Active,   QPalette.ColorRole.ButtonText,   Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorGroup.Inactive, QPalette.ColorRole.ButtonText,   Qt.GlobalColor.white)

    dark_palette.setColor(QPalette.ColorGroup.Active,   QPalette.ColorRole.Highlight,     highlight_color)
    dark_palette.setColor(QPalette.ColorGroup.Active,   QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    dark_palette.setColor(QPalette.ColorGroup.Inactive, QPalette.ColorRole.Highlight,     highlight_color)
    dark_palette.setColor(QPalette.ColorGroup.Inactive, QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)

    dark_palette.setColor(QPalette.ColorGroup.Active,   QPalette.ColorRole.ToolTipBase,  Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorGroup.Active,   QPalette.ColorRole.ToolTipText,  Qt.GlobalColor.white)

    dark_palette.setColor(QPalette.ColorGroup.Active,   QPalette.ColorRole.BrightText,   Qt.GlobalColor.red)

    QApplication.setPalette(dark_palette)
