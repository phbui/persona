from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QRadioButton, QButtonGroup
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt

class TrainingModeStep(QWidget):
    def __init__(self, wizard):
        super().__init__()
        self.wizard = wizard

        layout = QVBoxLayout()

        title_label = QLabel("Select Training Mode")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        self.human_feedback_radio = QRadioButton("Human Feedback Mode")
        self.auto_training_radio = QRadioButton("Auto Training Mode")

        self.mode_button_group = QButtonGroup()
        self.mode_button_group.addButton(self.human_feedback_radio)
        self.mode_button_group.addButton(self.auto_training_radio)

        self.continue_training_button = QPushButton("Continue")
        self.continue_training_button.setEnabled(False)
        self.continue_training_button.clicked.connect(self.finalize_selection)

        self.human_feedback_radio.toggled.connect(self.check_mode_selected)
        self.auto_training_radio.toggled.connect(self.check_mode_selected)

        layout.addWidget(self.human_feedback_radio)
        layout.addWidget(self.auto_training_radio)
        layout.addWidget(self.continue_training_button)

        self.setLayout(layout)

    def check_mode_selected(self):
        self.continue_training_button.setEnabled(self.human_feedback_radio.isChecked() or self.auto_training_radio.isChecked())

    def finalize_selection(self):
        self.wizard.training_mode = "human_feedback" if self.human_feedback_radio.isChecked() else "auto_training"
        print("RL Model Loaded:", self.wizard.rl_model)
        print("LLM Model Loaded:", self.wizard.llm)
        print("Training Mode Selected:", self.wizard.training_mode)
        self.wizard.close()