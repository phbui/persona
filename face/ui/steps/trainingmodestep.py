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
        self.novel_generation_radio = QRadioButton("Novel Generation")

        self.mode_button_group = QButtonGroup()
        self.mode_button_group.addButton(self.human_feedback_radio)
        self.mode_button_group.addButton(self.auto_training_radio)
        self.mode_button_group.addButton(self.novel_generation_radio)

        self.continue_training_button = QPushButton("Continue")
        self.continue_training_button.setEnabled(False)
        self.continue_training_button.clicked.connect(self.finalize_selection)

        self.human_feedback_radio.toggled.connect(self.check_mode_selected)
        self.auto_training_radio.toggled.connect(self.check_mode_selected)
        self.novel_generation_radio.toggled.connect(self.check_mode_selected)

        layout.addWidget(self.human_feedback_radio)
        layout.addWidget(self.auto_training_radio)
        layout.addWidget(self.novel_generation_radio)
        layout.addWidget(self.continue_training_button)

        self.setLayout(layout)

    def check_mode_selected(self):
        self.continue_training_button.setEnabled(self.human_feedback_radio.isChecked() 
                                                 or self.auto_training_radio.isChecked() 
                                                 or self.novel_generation_radio.isChecked())

    def finalize_selection(self):
        if self.human_feedback_radio.isChecked():
            self.wizard.training_mode = "human_feedback"  
        elif self.auto_training_radio.isChecked():
            self.wizard.training_mode = "auto_training"
        elif self.novel_generation_radio.isChecked():
            self.wizard.training_mode = "novel_generation"
        print("RL Model Loaded:", self.wizard.rl_model)
        print("LLM Model Loaded:", self.wizard.llm_model)
        print("Training Mode Selected:", self.wizard.training_mode)
        self.wizard.show_training_step()