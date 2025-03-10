from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton
)

class RankingStep(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        layout = QVBoxLayout()

        self.rank_faces_button = QPushButton("Rank Valid Faces")
        self.rank_faces_button.clicked.connect(self.parent.rank_valid_faces)
        layout.addWidget(self.rank_faces_button)

        self.submit_button = QPushButton("Submit Ranking")
        self.submit_button.clicked.connect(self.parent.ranking_done)
        layout.addWidget(self.submit_button)

        self.setLayout(layout)

    def enable_buttons(self):
        self.rank_faces_button.setEnabled(True)
        self.submit_button.setEnabled(True)
