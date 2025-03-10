from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QGridLayout, QCheckBox
)
from PyQt6.QtCore import Qt

class FaceSelectionUI(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        layout = QVBoxLayout()

        self.situation_label = QLabel("")
        layout.addWidget(self.situation_label)

        self.face_grid = QGridLayout()
        layout.addLayout(self.face_grid)

        self.rank_faces_button = QPushButton("Rank Valid Faces")
        self.rank_faces_button.clicked.connect(self.parent.rank_valid_faces)
        self.rank_faces_button.setEnabled(False)
        layout.addWidget(self.rank_faces_button)

        self.submit_button = QPushButton("Submit Ranking")
        self.submit_button.clicked.connect(self.parent.submit_ranking)
        self.submit_button.setEnabled(False)
        layout.addWidget(self.submit_button)

        self.setLayout(layout)

    def display_faces(self, generated_faces):
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

            valid_checkbox = QCheckBox("Valid")
            valid_checkbox.setChecked(True) 
            invalid_checkbox = QCheckBox("Vnvalid")

            valid_checkbox.stateChanged.connect(lambda _, v=valid_checkbox, b=invalid_checkbox, au=au_values: self.toggle_valid_invalid(v, b))
            invalid_checkbox.stateChanged.connect(lambda _, v=valid_checkbox, b=invalid_checkbox, au=au_values: self.toggle_valid_invalid(b, v))

            self.checkboxes.append((valid_checkbox, invalid_checkbox, au_values))
            self.parent.valid_faces.append(au_values)

            self.face_grid.addWidget(label, i // 5, (i % 5) * 3)
            self.face_grid.addWidget(valid_checkbox, i // 5, (i % 5) * 3 + 1)
            self.face_grid.addWidget(invalid_checkbox, i // 5, (i % 5) * 3 + 2)

        self.update_valid_invalid_faces() 

    def toggle_valid_invalid(self, checked_box, other_box):
        if checked_box.isChecked():
            other_box.setChecked(False)
        else:
            checked_box.setChecked(True)
        self.update_valid_invalid_faces()

    def update_valid_invalid_faces(self):
        self.parent.valid_faces = [au for v, b, au in self.checkboxes if v.isChecked()]
        self.parent.invalid_faces = [au for v, b, au in self.checkboxes if b.isChecked()]

        self.rank_faces_button.setEnabled(bool(self.parent.valid_faces))
        self.submit_button.setEnabled(bool(self.parent.valid_faces))
