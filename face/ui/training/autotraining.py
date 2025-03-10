from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt


class AutoTraining(QWidget):
    def __init__(self, wizard):
        super().__init__()
        self.wizard = wizard

        layout = QVBoxLayout()

        title_label = QLabel("Auto Training Mode")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        self.start_training_button = QPushButton("Start Training")
        self.start_training_button.clicked.connect(self.start_training)
        layout.addWidget(self.start_training_button)

        self.setLayout(layout)

    def start_training(self):
        print("Starting Auto Training...")
        # TODO: Implement automatic RL + LLM training
