import os
import json
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from feat.plotting import plot_face
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, 
    QTabWidget, QHBoxLayout, QComboBox
)
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtCore import Qt
from ai.manager_ppo import Manager_PPO
from ai.manager_extraction import Manager_Extraction 
from ai.manager_llm import Manager_LLM
from ui.dropzone import DropZone
from ui.draggableface import DraggableFace
from io import BytesIO

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
        self.setup_model_selection()
        self.rl_agent = self.load_rl_model()
        self.llm = self.load_llm_model()
        self.setup_human_feedback()
        self.setup_auto_training()

    def setup_model_selection(self):
        self.model_selection_layout = QHBoxLayout()
        self.rl_model_dropdown = QComboBox()
        self.rl_model_dropdown.addItem("Train New RL Model")
        self.rl_model_dropdown.addItems(self.get_model_list("models/rl"))
        self.rl_model_dropdown.currentTextChanged.connect(self.load_rl_model)
        self.model_selection_layout.addWidget(QLabel("Select RL Model:"))
        self.model_selection_layout.addWidget(self.rl_model_dropdown)

        self.llm_model_dropdown = QComboBox()
        self.llm_model_dropdown.addItem("Train New LLM Model")
        self.llm_model_dropdown.addItems(self.get_model_list("models/llm"))
        self.llm_model_dropdown.currentTextChanged.connect(self.load_llm_model)
        self.model_selection_layout.addWidget(QLabel("Select LLM Model:"))
        self.model_selection_layout.addWidget(self.llm_model_dropdown)

        self.layout.addLayout(self.model_selection_layout)

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
        if selected_model == "Train New LLM Model":
            return Manager_LLM()
        return Manager_LLM.load(os.path.join("models/llm", selected_model))

    def setup_human_feedback(self):
        layout = QVBoxLayout()
        self.situation_label = QLabel("Situation: Loading...")
        self.situation_label.setFont(QFont("Arial", 14))
        self.situation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.situation_label)

        self.face_list = QHBoxLayout()  
        self.drop_zones = []

        for i in range(1, 11):  
            drop_zone = DropZone(i)
            self.face_list.addWidget(drop_zone)
            self.drop_zones.append(drop_zone)

        layout.addLayout(self.face_list)

        self.submit_button = QPushButton("Submit Ranking")
        self.submit_button.clicked.connect(self.submit_ranking)
        layout.addWidget(self.submit_button)

        self.human_feedback_tab.setLayout(layout)
        self.load_new_situation()

    def setup_auto_training(self):
        layout = QVBoxLayout()
        self.auto_train_button = QPushButton("Start Automatic Training")
        self.auto_train_button.clicked.connect(self.auto_train)
        layout.addWidget(self.auto_train_button)
        self.auto_training_tab.setLayout(layout)

    def load_new_situation(self):
        situations_file = "data/situations.json"

        try:
            with open(situations_file, "r", encoding="utf-8") as file:
                data = json.load(file)
                situations = data.get("situations", [])  
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading situations.json: {e}")
            situations = [
                "Happy reaction to winning a prize",
                "Angry response to an insult",
                "Sadness after losing a pet"
            ]  

        if situations:
            self.situation_label.setText(f"Situation: {np.random.choice(situations)}")
            self.generated_faces = self.generate_faces()
            self.display_faces()
        else:
            print("No situations found in JSON!")

    def display_faces(self):
        for drop_zone in self.drop_zones:
            drop_zone.clear()  

        for i, au in enumerate(self.generated_faces):
            pixmap = self.generate_face_pixmap(au, size=(150, 150))  
            face_item = DraggableFace(au, pixmap)

            if i < len(self.drop_zones):
                self.drop_zones[i].set_face(face_item)  
            else:
                self.face_list.addItem(face_item)  

    def generate_face_pixmap(self, au_values, size=(120, 140)): 
        fig, ax = plt.subplots(figsize=(7, 8), dpi=300) 
        plot_face(ax=ax, au=au_values)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False) 
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=300, facecolor='white')  
        buf.seek(0)
        
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue(), 'PNG')
        buf.close()

        return pixmap.scaled(size[0], size[1], Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

    def generate_faces(self):
        text_situation = self.situation_label.text().replace("Situation: ", "")
        state = self.manager_extraction.extract_features(text_situation)
        state = th.tensor(state, dtype=th.float32).unsqueeze(0)
        state = state.to(next(self.rl_agent.policy.parameters()).device)

        generated_faces = []
        for _ in range(10):
            action, _, _ = self.rl_agent.policy.select_action(state)
            action_au = np.clip(action, 0, 3)
            generated_faces.append(action_au)

        return generated_faces

    def submit_ranking(self):
        ranking = []
        for drop_zone in self.drop_zones:
            if drop_zone.face:
                ranking.append((drop_zone.rank, drop_zone.face))

        ranking.sort()  
        ranked_faces = [face for _, face in ranking]

        self.train_rl_model(ranked_faces)
        self.finetune_llm(ranked_faces)
        self.load_new_situation()

    def train_rl_model(self, ranking):
        for rank, action_au in enumerate(ranking):
            reward = 1 - (rank / len(ranking))
            self.rl_agent.store_transition(state=np.random.rand(9), action=action_au, log_prob=0, reward=reward, value=0, done=False)

        if len(self.rl_agent.states) >= 10:
            self.rl_agent.update_policy()

    def finetune_llm(self, ranking):
        self.llm.finetune(ranking)

    def auto_train(self):
        text_situation = self.situation_label.text().replace("Situation: ", "")
        ranking_prompt = f"Rank the following faces for the situation: {text_situation}\n\n"
        for i, face in enumerate(self.generated_faces):
            ranking_prompt += f"{i+1}. AU Intensities: {face}\n"

        llm_ranking = self.llm.generate_response(ranking_prompt)
        ranking = [int(x) - 1 for x in llm_ranking.split(",")]
        self.train_rl_model(ranking)
