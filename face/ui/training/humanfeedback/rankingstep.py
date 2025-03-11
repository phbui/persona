from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon

class RankingStep(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        layout = QVBoxLayout()

        self.situation_label = QLabel("")
        layout.addWidget(self.situation_label)

        self.valid_faces_list = QListWidget()
        self.valid_faces_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.valid_faces_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        layout.addWidget(QLabel("Rank valid faces:"))
        layout.addWidget(self.valid_faces_list)

        self.submit_button = QPushButton("Submit Ranking")
        self.submit_button.clicked.connect(self.submit_ranking)
        self.submit_button.setEnabled(False)
        layout.addWidget(self.submit_button)

        self.setLayout(layout)

    def display_faces(self, valid_faces, situation):
        self.situation_label.setText(f"Situation: {situation}")

        self.valid_faces_list.clear()

        for face in valid_faces:
            pixmap = self.parent.generate_face_pixmap(face, size=(100, 100))  
            icon = QIcon(pixmap) 
            
            item = QListWidgetItem(f"Face {valid_faces.index(face) + 1}")
            item.setData(Qt.ItemDataRole.UserRole, face)
            item.setIcon(icon) 
            self.valid_faces_list.addItem(item)

        self.submit_button.setEnabled(bool(valid_faces))

    def submit_ranking(self):
        ranked_valid_faces = [self.valid_faces_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.valid_faces_list.count())]
        self.parent.valid_faces = ranked_valid_faces
        self.parent.ranking_done()
