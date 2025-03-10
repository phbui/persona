from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton
)
from PyQt6.QtCore import Qt

class RankingStep(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        layout = QVBoxLayout()

        title_label = QLabel("Rank Valid Faces")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        self.rank_faces_button = QPushButton("Rank Faces")
        self.rank_faces_button.clicked.connect(self.parent.rank_valid_faces)
        layout.addWidget(self.rank_faces_button)

        self.submit_button = QPushButton("Submit Ranking")
        self.submit_button.clicked.connect(self.parent.submit_ranking)
        layout.addWidget(self.submit_button)

        self.setLayout(layout)
        self.rank_faces_button.setEnabled(False)
        self.submit_button.setEnabled(False)

    def enable_buttons(self):
        self.rank_faces_button.setEnabled(True)
        self.submit_button.setEnabled(True)
