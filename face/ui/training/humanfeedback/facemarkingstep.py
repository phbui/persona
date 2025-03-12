from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QGridLayout, QToolButton, 
    QScrollArea, QFrame
)
from PyQt6.QtCore import Qt
from functools import partial
import numpy as np

class FaceMarkingStep(QWidget):
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
        self.face_grid = QGridLayout()
        self.scroll_layout.addLayout(self.face_grid)

        self.next_button = QPushButton("Proceed to next step")
        self.next_button.clicked.connect(self.parent.show_ranking_step)
        self.scroll_layout.addWidget(self.next_button)
        self.next_button.setEnabled(True)

        self.scroll_area.setWidget(self.scroll_widget)
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.scroll_area)
        self.setLayout(main_layout)

    def display_faces(self, generated_faces, situation, name, character_description):
        self.name_label.setText(f"Name: {name}")
        self.character_description_label.setText(f"Character Description: {character_description}")
        self.situation_label.setText(f"Situation: {situation}")

        while self.face_grid.count():
            item = self.face_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.checkboxes = []
        self.parent.valid_faces = []
        self.parent.invalid_faces = []

        for i, au_values in enumerate(generated_faces):
            pixmap = self.parent.generate_face_pixmap(au_values, size=(200, 200))

            label = QLabel()
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            toggle_button = QToolButton()
            toggle_button.setText("✔️ Valid")
            toggle_button.setCheckable(True)
            toggle_button.setChecked(True)
            toggle_button.setStyleSheet("background-color: lightgreen; font-weight: bold;")
            toggle_button.setFixedWidth(100)

            toggle_button.clicked.connect(partial(self.toggle_valid_invalid, toggle_button, au_values))

            self.checkboxes.append((toggle_button, au_values))
            self.parent.valid_faces.append(au_values)

            self.face_grid.addWidget(label, i // 5, (i % 5) * 2)
            self.face_grid.addWidget(toggle_button, i // 5, (i % 5) * 2 + 1)

    def toggle_valid_invalid(self, button, au_values):
        if button.isChecked():
            button.setText("✔️ Valid")
            button.setStyleSheet("background-color: lightgreen; font-weight: bold;")
            if not any(np.array_equal(x, au_values) for x in self.parent.valid_faces):
                self.parent.valid_faces.append(au_values)
            self.parent.invalid_faces = [x for x in self.parent.invalid_faces if not np.array_equal(x, au_values)]
        else:
            button.setText("❌ Invalid")
            button.setStyleSheet("background-color: lightcoral; font-weight: bold;")
            if not any(np.array_equal(x, au_values) for x in self.parent.invalid_faces):
                self.parent.invalid_faces.append(au_values)
            self.parent.valid_faces = [x for x in self.parent.valid_faces if not np.array_equal(x, au_values)]
