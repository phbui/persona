from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QSpinBox
)
from PyQt6.QtCore import Qt

class EpochSelectionStep(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        layout = QVBoxLayout()

        title_label = QLabel("Select Number of Epochs")
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
        self.parent.show_face_marking_step()
