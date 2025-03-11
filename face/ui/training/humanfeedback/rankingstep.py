from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QListWidget, QListWidgetItem, QHBoxLayout
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

        main_layout = QHBoxLayout()  

        self.valid_faces_list = QListWidget()
        self.valid_faces_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        main_layout.addWidget(self.valid_faces_list)

        button_layout = QVBoxLayout() 

        self.move_up_button = QPushButton("↑")
        self.move_down_button = QPushButton("↓")

        self.move_up_button.clicked.connect(self.move_selected_up)
        self.move_down_button.clicked.connect(self.move_selected_down)

        button_layout.addWidget(self.move_up_button)
        button_layout.addWidget(self.move_down_button)

        main_layout.addLayout(button_layout)

        layout.addWidget(QLabel("Rank valid faces:"))
        layout.addLayout(main_layout)

        self.submit_button = QPushButton("Submit Ranking")
        self.submit_button.clicked.connect(self.submit_ranking)
        self.submit_button.setEnabled(False)
        layout.addWidget(self.submit_button)

        self.setLayout(layout)

    def display_faces(self, valid_faces, situation):
        self.situation_label.setText(f"Situation: {situation}")
        self.valid_faces_list.clear()
        
        self.face_data = valid_faces[:] 
        self.render_faces()

        self.submit_button.setEnabled(bool(valid_faces))

    def render_faces(self):
        self.valid_faces_list.clear()
        for face in self.face_data:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, face)
            pixmap = self.parent.generate_face_pixmap(face, size=(150, 150))

            icon = QIcon(pixmap)
            item.setIcon(icon)

            item.setSizeHint(pixmap.size())  
            self.valid_faces_list.addItem(item)

    def move_selected_up(self):
        current_index = self.valid_faces_list.currentRow()
        if current_index > 0:
            self.face_data[current_index], self.face_data[current_index - 1] = (
                self.face_data[current_index - 1], self.face_data[current_index]
            )
            self.render_faces()
            self.valid_faces_list.setCurrentRow(current_index - 1)

    def move_selected_down(self):
        current_index = self.valid_faces_list.currentRow()
        if current_index < len(self.face_data) - 1:
            self.face_data[current_index], self.face_data[current_index + 1] = (
                self.face_data[current_index + 1], self.face_data[current_index]
            )
            self.render_faces()
            self.valid_faces_list.setCurrentRow(current_index + 1)  

    def submit_ranking(self):
        self.parent.valid_faces = self.face_data
        self.parent.ranking_done()
