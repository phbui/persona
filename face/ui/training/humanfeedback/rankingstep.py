from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QListWidget, QListWidgetItem, 
    QHBoxLayout, QScrollArea, QFrame
)
from PyQt6.QtCore import Qt
import numpy as np

class RankingStep(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setGeometry(10, 10, 800, 1000)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)  

        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)

        self.name_label = QLabel("")
        self.name_label.setWordWrap(True)
        self.scroll_layout.addWidget(self.name_label)

        self.character_description_label = QLabel("")
        self.character_description_label.setWordWrap(True)
        self.scroll_layout.addWidget(self.character_description_label)

        self.situation_label = QLabel("")
        self.situation_label.setWordWrap(True)
        self.scroll_layout.addWidget(self.situation_label)

        main_layout = QHBoxLayout()
        self.valid_faces_list = QListWidget()
        self.valid_faces_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        main_layout.addWidget(self.valid_faces_list)

        button_layout = QVBoxLayout()
        self.go_back_button = QPushButton("Go Back") 
        self.move_up_button = QPushButton("↑")
        self.move_down_button = QPushButton("↓")

        self.go_back_button.clicked.connect(self.go_back_to_facemarking) 
        self.move_up_button.clicked.connect(self.move_selected_up)
        self.move_down_button.clicked.connect(self.move_selected_down)

        button_layout.addWidget(self.go_back_button) 
        button_layout.addWidget(self.move_up_button)
        button_layout.addWidget(self.move_down_button)
        main_layout.addLayout(button_layout)

        self.scroll_layout.addWidget(QLabel("Rank valid faces:"))
        self.scroll_layout.addLayout(main_layout)

        self.submit_button = QPushButton("Submit Ranking")
        self.submit_button.clicked.connect(self.submit_ranking)
        self.submit_button.setEnabled(False)
        self.scroll_layout.addWidget(self.submit_button)

        self.scroll_area.setWidget(self.scroll_widget)
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.scroll_area)
        self.setLayout(main_layout)

        self.face_pixmaps = {}

    def display_faces(self, valid_faces, situation, name, character_description):
        self.name_label.setText(f"Name: {name}")
        self.character_description_label.setText(f"Character Description: {character_description}")
        self.situation_label.setText(f"Situation: {situation}")
        self.valid_faces_list.clear()

        self.face_data = valid_faces[:]
        self.face_pixmaps.clear()

        for face in valid_faces:
            face_key = tuple(face.tolist()) 
            if face_key not in self.face_pixmaps:
                pixmap = self.parent.generate_face_pixmap(face['aus'], size=(150, 150))
                self.face_pixmaps[face_key] = pixmap

        self.render_faces()
        self.submit_button.setEnabled(bool(valid_faces))

    def render_faces(self):
        self.valid_faces_list.clear()
        for face in self.face_data:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, face)

            face_key = tuple(face.tolist()) 
            pixmap = self.face_pixmaps.get(face_key)

            if pixmap:
                widget = QWidget()
                layout = QHBoxLayout()

                label = QLabel()
                label.setPixmap(pixmap)
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(label)

                widget.setLayout(layout)
                item.setSizeHint(widget.sizeHint())

                self.valid_faces_list.addItem(item)
                self.valid_faces_list.setItemWidget(item, widget)

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

    def go_back_to_facemarking(self):
        self.parent.go_back_to_face_marking_step()

    def submit_ranking(self):
        self.parent.valid_faces = [np.array(face) for face in self.face_data]
        self.parent.ranking_done()
