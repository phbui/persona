import json
import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt


class CharacterDescriptionStep(QWidget):
    def __init__(self, wizard):
        super().__init__()
        self.wizard = wizard

        layout = QVBoxLayout()

        title_label = QLabel("Select Character Description")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        self.description_dropdown = QComboBox()
        self.characters = self.load_character_descriptions()
        for name, desc in self.characters.items():
            self.description_dropdown.addItem(f"{name}: {desc[:50]}...", (name, desc))  

        layout.addWidget(self.description_dropdown)

        self.continue_button = QPushButton("Next")
        self.continue_button.clicked.connect(self.proceed)
        layout.addWidget(self.continue_button)

        self.setLayout(layout)

    def load_character_descriptions(self):
        file_path = "data/character_descriptions.json"
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    return {char["name"]: char["description"] for char in data.get("character_descriptions", [])}
            except (json.JSONDecodeError, KeyError):
                return {"Error": "Failed to load descriptions"}
        return {"No Data": "No character descriptions found"}

    def proceed(self):
        selected_data = self.description_dropdown.currentData()
        if selected_data:
            name, full_description = selected_data
            self.wizard.name = name
            self.wizard.character_description = full_description

        else:
            self.wizard.character_description = "No description selected."

        self.wizard.show_model_selection_step()
