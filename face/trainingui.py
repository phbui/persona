import numpy as np
import matplotlib.pyplot as plt
from feat.plotting import plot_face
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, 
    QTabWidget, QListWidget, QListWidgetItem, QHBoxLayout
)
from PyQt6.QtGui import QPixmap, QFont
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
        self.llm = Manager_LLM()

        self.tabs.addTab(self.human_feedback_tab, "Human Feedback")
        self.tabs.addTab(self.auto_training_tab, "Automatic Training")
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

        self.manager_extraction = Manager_Extraction()

        self.rl_agent = Manager_PPO(input_dim=9, num_candidates=20)


    def setup_human_feedback(self):
        layout = QVBoxLayout()

        # Display Situation
        self.situation_label = QLabel("Situation: Loading...")
        self.situation_label.setFont(QFont("Arial", 14))
        self.situation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.situation_label)

        # Face display area
        self.face_layout = QHBoxLayout()
        self.face_widgets = []
        for _ in range(10):  # 10 generated faces
            face_canvas = FigureCanvas(plt.figure(figsize=(4, 5)))
            self.face_widgets.append(face_canvas)
            self.face_layout.addWidget(face_canvas)
        layout.addLayout(self.face_layout)

        # Ranking list
        self.face_list = QListWidget()
        self.face_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        layout.addWidget(self.face_list)

        # Submit Button
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

        # Generate AUs using RL
        self.generated_faces = self.generate_faces()
        self.display_faces()
                    
    def generate_faces(self):
        """Uses RL system to generate 10 AU sets where each value is between [0,3]."""
        text_situation = self.situation_label.text().replace("Situation: ", "")
        state = self.manager_extraction.extract_features(text_situation)

        generated_faces = []
        for _ in range(10):
            action, _, _ = self.rl_agent.policy.select_action(state)
            action_au = np.clip(action, 0, 3)  # Ensure AU values are in range
            generated_faces.append(action_au)

        return generated_faces  

    def display_faces(self):
        """Plots AU-generated faces using Py-Feat and displays them in the UI."""
        self.face_list.clear()
        for i, (face_canvas, au) in enumerate(zip(self.face_widgets, self.generated_faces)):
            face_canvas.figure.clear()
            ax = face_canvas.figure.add_subplot(111)
            plot_face(ax=ax, au=au, title=f"Face {i+1}")
            face_canvas.draw()

            item = QListWidgetItem(f"Face {i+1}")
            self.face_list.addItem(item)

    def submit_ranking(self):
        """Collects ranking input and trains the RL system."""
        ranking = [self.face_list.row(self.face_list.item(i)) for i in range(self.face_list.count())]
        print("User Ranking Submitted:", ranking)

        self.train_models(ranking)
        self.load_new_situation()


    def train_models(self, ranking):
        """Trains RL system with user rankings."""
        print("Training RL system with user feedback...")
        for rank, action_au in enumerate(self.generated_faces):
            reward = 1 - (rank / len(self.generated_faces))  # Higher rank = higher reward
            self.rl_agent.store_transition(state=np.random.rand(100), action=action_au, log_prob=0, reward=reward, value=0, done=False)

        # Perform a training step every 10 rankings
        if len(self.rl_agent.states) >= 10:
            self.rl_agent.update_policy()

    def setup_auto_training(self):
        """Automated training using LLM ranking."""
        layout = QVBoxLayout()
        self.auto_train_button = QPushButton("Start Automatic Training")
        self.auto_train_button.clicked.connect(self.auto_train)
        layout.addWidget(self.auto_train_button)
        self.auto_training_tab.setLayout(layout)

    def auto_train(self):
        """Uses the LLM to rank AU faces and trains RL automatically."""
        print("Starting automatic training...")

        text_situation = self.situation_label.text().replace("Situation: ", "")

        state = self.manager_extraction.extract_features(text_situation)

        generated_faces = []
        for _ in range(10):
            action, _, _ = self.rl_agent.policy.select_action(state)
            action_au = np.clip(action, 0, 3)  # Ensure AU values are in range
            generated_faces.append(action_au)

        ranking_prompt = f"Rank the following faces based on their appropriateness for the situation: {text_situation}\n\n"
        for i, face in enumerate(generated_faces):
            ranking_prompt += f"{i+1}. AU Intensities: {face}\n"

        llm_ranking = self.llm.generate_response(ranking_prompt)

        # ✅ Convert ranking response to list
        ranking = [int(x) - 1 for x in llm_ranking.split(",")]

        # ✅ Assign rewards based on LLM ranking
        for rank, action_au in enumerate(generated_faces):
            reward = 1 - (rank / len(generated_faces))  # Higher rank = higher reward
            self.rl_agent.store_transition(state, action_au, log_prob=0, reward=reward, value=0, done=False)

        # ✅ Perform a training step
        self.rl_agent.update_policy()

