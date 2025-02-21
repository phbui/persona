from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt
from ..meta.meta_singleton import Meta_Singleton

class Interface(QWidget):
    """
    A base class for all tab interfaces that provides a common look & feel.
    Each derived class can supply its own title and description (or override any methods as needed).
    """

    def __init__(self, title: str, description: str):
        super().__init__()
        self.title = title
        self.description = description
        self._init_ui()

    def _init_ui(self):
        """
        Create a common layout and styling for all derived interfaces.
        """
        layout = QVBoxLayout()

        title_label = QLabel(self.title)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title_label)

        desc_label = QLabel(self.description)
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        self.setLayout(layout)

    # You could define additional common methods here, for example:
    def common_method(self):
        """
        A method that derived classes can use or override, if needed.
        """
        print("This is a method defined in the base interface class.")
