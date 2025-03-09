import os
import numpy as np
import matplotlib.pyplot as plt
from feat.plotting import plot_face
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, 
    QTabWidget, QListWidget, QListWidgetItem, QHBoxLayout, QComboBox
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from ai.manager_ppo import Manager_PPO  # RL system
from ai.manager_extraction import Manager_Extraction 
from ai.manager_llm import Manager_LLM

class TrainingUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Training Interface")
        self.setGeometry(100, 100, 1200, 800)

        self.layout = QVBoxLayout()
        self.tabs = QTabWidget()
        self.human_feedback_tab = QWidget()
        self.auto_training_tab = QWidget()

        self.tabs.addTab(self.human_feedback_tab, "Human Feedback")
        self.tabs.addTab(self.auto_training_tab, "Automatic Training")
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

        self.manager_extraction = Manager_Extraction()

        # 🔹 UI for Model Selection
        self.setup_model_selection()

        # 🔹 Initialize RL and LLM Models
        self.rl_agent = self.load_rl_model()
        self.llm = self.load_llm_model()

        # 🔹 Setup Training UI
        self.setup_human_feedback()
        self.setup_auto_training()

    def setup_model_selection(self):
        """Allows selection of RL and LLM models from saved models."""
        self.model_selection_layout = QHBoxLayout()
        
        # RL Model Selection
        self.rl_model_dropdown = QComboBox()
        self.rl_model_dropdown.addItem("Train New RL Model")
        self.rl_model_dropdown.addItems(self.get_model_list("models/rl"))
        self.rl_model_dropdown.currentTextChanged.connect(self.load_rl_model)
        self.model_selection_layout.addWidget(QLabel("Select RL Model:"))
        self.model_selection_layout.addWidget(self.rl_model_dropdown)

        # LLM Model Selection
        self.llm_model_dropdown = QComboBox()
        self.llm_model_dropdown.addItem("Train New LLM Model")
        self.llm_model_dropdown.addItems(self.get_model_list("models/llm"))
        self.llm_model_dropdown.currentTextChanged.connect(self.load_llm_model)
        self.model_selection_layout.addWidget(QLabel("Select LLM Model:"))
        self.model_selection_layout.addWidget(self.llm_model_dropdown)

        self.layout.addLayout(self.model_selection_layout)

    def get_model_list(self, folder):
        """Returns a list of saved models in the given folder."""
        if not os.path.exists(folder):
            os.makedirs(folder)
        return [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]

    def load_rl_model(self):
        """Loads a saved RL model or initializes a new one."""
        selected_model = self.rl_model_dropdown.currentText()
        if selected_model == "Train New RL Model":
            print("Initializing new RL model...")
            return Manager_PPO(input_dim=9, num_candidates=80)  # Updated to 80
        else:
            print(f"Loading RL model: {selected_model}")
            return Manager_PPO.load(os.path.join("models/rl", selected_model))  # Load method must be implemented in Manager_PPO

    def load_llm_model(self):
        """Loads a saved LLM model or initializes a new one."""
        selected_model = self.llm_model_dropdown.currentText()
        if selected_model == "Train New LLM Model":
            print("Initializing new LLM model...")
            return Manager_LLM()
        else:
            print(f"Loading LLM model: {selected_model}")
            return Manager_LLM.load(os.path.join("models/llm", selected_model))  # Load method must be implemented in Manager_LLM

    def setup_human_feedback(self):
        """Setup human feedback tab."""
        layout = QVBoxLayout()
        self.situation_label = QLabel("Situation: Loading...")
        self.situation_label.setFont(QFont("Arial", 14))
        self.situation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.situation_label)

        self.face_layout = QHBoxLayout()
        self.face_widgets = []
        for _ in range(10):  # 10 generated faces
            face_canvas = FigureCanvas(plt.figure(figsize=(4, 5)))
            self.face_widgets.append(face_canvas)
            self.face_layout.addWidget(face_canvas)
        layout.addLayout(self.face_layout)

        self.face_list = QListWidget()
        self.face_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        layout.addWidget(self.face_list)

        self.submit_button = QPushButton("Submit Ranking")
        self.submit_button.clicked.connect(self.submit_ranking)
        layout.addWidget(self.submit_button)

        self.human_feedback_tab.setLayout(layout)
        self.load_new_situation()

    def load_new_situation(self):
        """Loads a new situation and generates AU faces using RL system."""
        situations = [
            "Happy reaction to winning a prize",
            "Angry response to an insult",
            "Sadness after losing a pet"
        ]
        self.situation_label.setText(f"Situation: {np.random.choice(situations)}")

        self.generated_faces = self.generate_faces()
        self.display_faces()

    def generate_faces(self):
        """Uses RL system to generate 10 AU sets where each value is between [0,3]."""
        text_situation = self.situation_label.text().replace("Situation: ", "")
        state = self.manager_extraction.extract_features(text_situation)

        generated_faces = []
        for _ in range(10):
            action, _, _ = self.rl_agent.policy.select_action(state)
            action_au = np.clip(action, 0, 3)
            generated_faces.append(action_au)

        return generated_faces  

    def submit_ranking(self):
        """Trains both RL and LLM using human rankings."""
        ranking = [self.face_list.row(self.face_list.item(i)) for i in range(self.face_list.count())]

        # Train RLHF
        self.train_rl_model(ranking)
        # Train LLM using RLHF
        self.finetune_llm(ranking)

        self.load_new_situation()

    def train_rl_model(self, ranking):
        """Trains RL system with user rankings."""
        for rank, action_au in enumerate(self.generated_faces):
            reward = 1 - (rank / len(self.generated_faces))
            self.rl_agent.store_transition(state=np.random.rand(9), action=action_au, log_prob=0, reward=reward, value=0, done=False)

        if len(self.rl_agent.states) >= 10:
            self.rl_agent.update_policy()

    def finetune_llm(self, ranking):
        """Finetunes the LLM based on human rankings."""
        self.llm.finetune(ranking)  # Implement this in Manager_LLM

    def auto_train(self):
        """Uses the finetuned LLM to rank AU faces and trains RL automatically."""
        text_situation = self.situation_label.text().replace("Situation: ", "")
        ranking = self.llm.generate_response(f"Rank these faces: {text_situation}")
        self.train_rl_model(ranking)

