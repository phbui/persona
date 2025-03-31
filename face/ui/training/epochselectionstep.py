from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QSpinBox
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt

class EpochSelectionStep(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        layout = QVBoxLayout()

        title_label = QLabel("Select Number of Epochs")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        self.epoch_selector = QSpinBox()
        self.epoch_selector.setRange(1, 100)
        self.epoch_selector.setValue(1)
        layout.addWidget(self.epoch_selector)

        self.confirm_button = QPushButton("Confirm Epochs")
        self.confirm_button.clicked.connect(self.confirm_epochs)
        layout.addWidget(self.confirm_button)

        self.setLayout(layout)

    def confirm_epochs(self):
        self.parent.epochs = self.epoch_selector.value()
        print(f"Selected {self.parent.epochs} epochs")
        self.parent.post_epoch_step()
