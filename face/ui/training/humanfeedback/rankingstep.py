from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel
)

class RankingStep(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        layout = QVBoxLayout()

        self.situation_label = QLabel("")
        layout.addWidget(self.situation_label)  

        self.rank_faces_button = QPushButton("Rank Faces")
        self.rank_faces_button.clicked.connect(self.parent.rank_valid_faces)
        layout.addWidget(self.rank_faces_button)
        self.rank_faces_button.setEnabled(False)

        self.submit_button = QPushButton("Submit Ranking")
        self.submit_button.clicked.connect(self.parent.ranking_done)
        layout.addWidget(self.submit_button)
        self.submit_button.setEnabled(False)

        self.setLayout(layout)

    def display_situation(self, situation):
        """Update situation label in ranking step."""
        self.situation_label.setText(f"Situation: {situation}")
