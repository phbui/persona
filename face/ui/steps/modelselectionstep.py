import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QLineEdit
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt
from ai.manager_ppo import Manager_PPO
from ai.manager_llm import Manager_LLM


class ModelSelectionStep(QWidget):
    def __init__(self, wizard):
        super().__init__()
        self.wizard = wizard

        layout = QVBoxLayout()

        title_label = QLabel("Select Models Before Proceeding")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        self.rl_model_dropdown = QComboBox()
        self.rl_model_dropdown.addItem("Train New RL Model")
        self.rl_model_dropdown.addItems(self.get_model_list("models/rl"))

        self.rl_model_name_input = QLineEdit()
        self.rl_model_name_input.setPlaceholderText("Enter new RL model name...")
        self.rl_model_name_input.setVisible(False)

        self.llm_model_dropdown = QComboBox()
        self.llm_model_dropdown.addItem("Finetune New LLM Model")
        self.llm_model_dropdown.addItems(self.get_model_list("models/llm"))

        self.llm_model_name_input = QLineEdit()
        self.llm_model_name_input.setPlaceholderText("Enter new LLM model name...")
        self.llm_model_name_input.setVisible(False)

        self.continue_button = QPushButton("Next")
        self.continue_button.setEnabled(False)
        self.continue_button.clicked.connect(self.proceed)

        self.rl_model_dropdown.currentTextChanged.connect(self.update_model_name_input)
        self.llm_model_dropdown.currentTextChanged.connect(self.update_model_name_input)
        self.rl_model_name_input.textChanged.connect(self.check_ready)
        self.llm_model_name_input.textChanged.connect(self.check_ready)

        layout.addWidget(QLabel("Select RL Model:"))
        layout.addWidget(self.rl_model_dropdown)
        layout.addWidget(self.rl_model_name_input)
        layout.addWidget(QLabel("Select LLM Model:"))
        layout.addWidget(self.llm_model_dropdown)
        layout.addWidget(self.llm_model_name_input)
        layout.addWidget(self.continue_button)

        self.setLayout(layout)
        self.update_model_name_input()

    def get_model_list(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        return [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]

    def update_model_name_input(self):
        self.rl_model_name_input.setVisible(self.rl_model_dropdown.currentText() == "Train New RL Model")
        self.llm_model_name_input.setVisible(self.llm_model_dropdown.currentText() == "Finetune New LLM Model")
        self.check_ready()

    def check_ready(self):
        rl_selected = self.rl_model_dropdown.currentText() != "Train New RL Model" or bool(self.rl_model_name_input.text().strip())
        llm_selected = self.llm_model_dropdown.currentText() != "Finetune New LLM Model" or bool(self.llm_model_name_input.text().strip())
        self.continue_button.setEnabled(rl_selected and llm_selected)

    def proceed(self):
        self.wizard.rl_model = self.load_rl_model()
        self.wizard.llm_model = self.load_llm_model()
        self.wizard.show_training_mode_step()

    def load_rl_model(self):
        selected_model = self.rl_model_dropdown.currentText()
        if selected_model == "Train New RL Model":
            model_name = self.rl_model_name_input.text().strip()
            model_path = os.path.join("models/rl", f"{model_name}.pth")
            self.wizard.rl_model_path = model_path
            return Manager_PPO(input_dim=9, action_dim=20, num_categories=4, model_path=model_path)
        
        model_path = os.path.join("models/rl", f"{selected_model}.pth")
        self.wizard.rl_model_path = model_path
        return Manager_PPO.load(model_path)

    def load_llm_model(self):
        selected_model = self.llm_model_dropdown.currentText()
        if selected_model == "Finetune New LLM Model":
            model_name = self.llm_model_name_input.text().strip()
            model_path = os.path.join("models/llm", model_name)
            self.wizard.llm_model_path = model_path
            return Manager_LLM(model_path=model_path)
        model_path = os.path.join("models/llm", selected_model)
        self.wizard.llm_model_path = model_path
        return Manager_LLM.load(model_path)