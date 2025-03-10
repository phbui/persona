import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, 
    QComboBox, QStackedWidget
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt
from ai.manager_ppo import Manager_PPO
from ai.manager_llm import Manager_LLM

class TrainingUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Model Selection Wizard")
        self.setGeometry(100, 100, 800, 400)

        self.layout = QVBoxLayout()
        self.stacked_widget = QStackedWidget()
        self.layout.addWidget(self.stacked_widget)
        self.setLayout(self.layout)

        self.rl_model = None
        self.llm = None

        self.setup_wizard_ui()

    def setup_wizard_ui(self):
        self.model_selection_widget = QWidget()
        layout = QVBoxLayout()

        title_label = QLabel("Select Models Before Proceeding")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        self.rl_model_dropdown = QComboBox()
        self.rl_model_dropdown.addItem("Train New RL Model")
        self.rl_model_dropdown.addItems(self.get_model_list("models/rl"))

        self.llm_model_dropdown = QComboBox()
        self.llm_model_dropdown.addItem("Finetune New LLM Model")
        self.llm_model_dropdown.addItems(self.get_model_list("models/llm"))

        self.continue_button = QPushButton("Continue")
        self.continue_button.setEnabled(False)
        self.continue_button.clicked.connect(self.initialize_models)

        self.rl_model_dropdown.currentTextChanged.connect(self.check_ready)
        self.llm_model_dropdown.currentTextChanged.connect(self.check_ready)

        layout.addWidget(QLabel("Select RL Model:"))
        layout.addWidget(self.rl_model_dropdown)
        layout.addWidget(QLabel("Select LLM Model:"))
        layout.addWidget(self.llm_model_dropdown)
        layout.addWidget(self.continue_button)

        self.model_selection_widget.setLayout(layout)
        self.stacked_widget.addWidget(self.model_selection_widget)

        self.check_ready()

    def check_ready(self):
        if self.rl_model_dropdown.currentText() and self.llm_model_dropdown.currentText():
            self.continue_button.setEnabled(True)
        else:
            self.continue_button.setEnabled(False)


    def initialize_models(self):
        self.rl_model = self.load_rl_model()
        self.llm = self.load_llm_model()
        print("RL Model Loaded:", self.rl_model)
        print("LLM Model Loaded:", self.llm)
        self.close()

    def get_model_list(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        return [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]

    def load_rl_model(self):
        selected_model = self.rl_model_dropdown.currentText()
        if selected_model == "Train New RL Model":
            return Manager_PPO(input_dim=9, action_dim=20, num_categories=4)
        return Manager_PPO.load(os.path.join("models/rl", selected_model))

    def load_llm_model(self):
        selected_model = self.llm_model_dropdown.currentText()
        if selected_model == "Finetune New LLM Model":
            return Manager_LLM()
        return Manager_LLM.load(os.path.join("models/llm", selected_model))
