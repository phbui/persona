from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QPixmap
import numpy as np  

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
        
        for i, face in enumerate(valid_faces):
            pixmap = self.parent.generate_face_pixmap(face, size=(150, 150))  
            widget = QWidget()
            layout = QVBoxLayout()
            
            label = QLabel()
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            layout.addWidget(label)
            widget.setLayout(layout)

            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, face)
            item.setSizeHint(pixmap.size()) 
            
            self.valid_faces_list.addItem(item)
            self.valid_faces_list.setItemWidget(item, widget)
            
        self.submit_button.setEnabled(bool(valid_faces))

    def submit_ranking(self):
        ranked_valid_faces = [self.valid_faces_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.valid_faces_list.count())]
        self.parent.valid_faces = ranked_valid_faces
        self.parent.ranking_done()
