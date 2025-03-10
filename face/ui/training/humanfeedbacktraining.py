import os
import json
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from feat.plotting import plot_face
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QListWidget, QListWidgetItem,
    QHBoxLayout, QSpinBox
)
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtCore import Qt
from ai.manager_ppo import Manager_PPO
from ai.manager_extraction import Manager_Extraction
from ai.manager_llm import Manager_LLM
from io import BytesIO


class HumanFeedbackTraining(QWidget):
    def __init__(self, wizard):
        super().__init__()
        self.wizard = wizard
        self.rl_agent = wizard.rl_model
        self.llm = wizard.llm
        self.manager_extraction = Manager_Extraction()
        self.epochs = None  
        self.situations = self.load_situations()

        layout = QVBoxLayout()

        title_label = QLabel("Human Feedback Training")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        self.epoch_selector = QSpinBox()
        self.epoch_selector.setRange(1, 100)
        self.epoch_selector.setValue(1)
        layout.addWidget(QLabel("Select Number of Epochs:"))
        layout.addWidget(self.epoch_selector)

        self.confirm_epochs_button = QPushButton("Confirm Epochs")
        self.confirm_epochs_button.clicked.connect(self.confirm_epochs)
        layout.addWidget(self.confirm_epochs_button)

        self.start_training_button = QPushButton("Start Training")
        self.start_training_button.setEnabled(False)  
        self.start_training_button.clicked.connect(self.start_training)
        layout.addWidget(self.start_training_button)

        self.situation_label = QLabel("")
        layout.addWidget(self.situation_label)

        self.face_list = QListWidget()
        self.face_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        layout.addWidget(self.face_list)

        self.mark_valid_button = QPushButton("Mark as Valid")
        self.mark_valid_button.clicked.connect(self.mark_valid)
        self.mark_valid_button.setEnabled(False)
        layout.addWidget(self.mark_valid_button)

        self.mark_bad_button = QPushButton("Mark as Bad")
        self.mark_bad_button.clicked.connect(self.mark_bad)
        self.mark_bad_button.setEnabled(False)
        layout.addWidget(self.mark_bad_button)

        self.rank_faces_button = QPushButton("Rank Valid Faces")
        self.rank_faces_button.clicked.connect(self.rank_valid_faces)
        self.rank_faces_button.setEnabled(False)
        layout.addWidget(self.rank_faces_button)

        self.submit_button = QPushButton("Submit Ranking")
        self.submit_button.clicked.connect(self.submit_ranking)
        self.submit_button.setEnabled(False)
        layout.addWidget(self.submit_button)

        self.setLayout(layout)

        self.current_epoch = 0
        self.current_situation_index = 0
        self.generated_faces = []
        self.valid_faces = []
        self.bad_faces = []

    def confirm_epochs(self):
        self.epochs = self.epoch_selector.value()
        self.epoch_selector.setEnabled(False)
        self.confirm_epochs_button.setEnabled(False)
        self.start_training_button.setEnabled(True)  

    def load_situations(self):
        situations_file = "data/situations.json"
        try:
            with open(situations_file, "r", encoding="utf-8") as file:
                data = json.load(file)
                return data.get("situations", [])
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading situations.json: {e}")
            return ["Happy reaction to winning a prize", "Angry response to an insult", "Sadness after losing a pet"]

    def start_training(self):
        self.current_epoch = 0
        self.current_situation_index = 0
        self.run_epoch()

    def run_epoch(self):
        if self.current_epoch >= self.epochs:
            print("Training complete.")
            return

        self.current_situation_index = 0
        self.process_next_situation()

    def process_next_situation(self):
        if self.current_situation_index >= len(self.situations):
            self.current_epoch += 1
            self.save_models()  
            self.run_epoch()
            return

        situation = self.situations[self.current_situation_index]
        self.situation_label.setText(f"Situation: {situation}")

        state = self.manager_extraction.extract_features(situation)
        self.generated_faces = self.generate_faces(state)

        self.display_faces()
        self.valid_faces = []
        self.bad_faces = []

        self.mark_valid_button.setEnabled(True)
        self.mark_bad_button.setEnabled(True)
        self.rank_faces_button.setEnabled(False)
        self.submit_button.setEnabled(False)

    def generate_faces(self, state):
        state_tensor = th.tensor(state, dtype=th.float32).unsqueeze(0)
        state_tensor = state_tensor.to(next(self.rl_agent.policy.parameters()).device)

        faces = []
        for _ in range(10):
            action, _, _ = self.rl_agent.policy.select_action(state_tensor)
            action_au = np.clip(action, 0, 3)
            faces.append(action_au)

        return faces

    def display_faces(self):
        self.face_list.clear()

        for i, au_values in enumerate(self.generated_faces):
            pixmap = self.generate_face_pixmap(au_values)
            item = QListWidgetItem(f"Face {i+1}")
            item.setData(Qt.ItemDataRole.UserRole, au_values)
            icon = QPixmap(pixmap)
            item.setIcon(icon)
            self.face_list.addItem(item)

    def generate_face_pixmap(self, au_values, size=(150, 150)):
        fig, ax = plt.subplots(figsize=(7, 8), dpi=300)
        plot_face(ax=ax, au=au_values)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=300, facecolor="white")
        buf.seek(0)

        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue(), "PNG")
        buf.close()

        return pixmap.scaled(size[0], size[1], Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

    def mark_valid(self):
        selected_items = self.face_list.selectedItems()
        for item in selected_items:
            au_values = item.data(Qt.ItemDataRole.UserRole)
            if au_values not in self.valid_faces:
                self.valid_faces.append(au_values)

        self.check_marking_done()

    def mark_bad(self):
        selected_items = self.face_list.selectedItems()
        for item in selected_items:
            au_values = item.data(Qt.ItemDataRole.UserRole)
            if au_values not in self.bad_faces:
                self.bad_faces.append(au_values)

        self.check_marking_done()

    def check_marking_done(self):
        if self.valid_faces or self.bad_faces:
            self.rank_faces_button.setEnabled(True)

    def rank_valid_faces(self):
        self.valid_faces.sort(key=lambda x: self.valid_faces.index(x))
        self.submit_button.setEnabled(True)

    def submit_ranking(self):
        self.train_rl_model()
        self.finetune_llm()

        self.current_situation_index += 1
        self.process_next_situation()

    def train_rl_model(self):
        for action_au in self.bad_faces:
            self.rl_agent.store_transition(state=np.random.rand(9), action=action_au, log_prob=0, reward=-1, value=0, done=False)

        for rank, action_au in enumerate(self.valid_faces):
            reward = 1 - (rank / len(self.valid_faces))
            self.rl_agent.store_transition(state=np.random.rand(9), action=action_au, log_prob=0, reward=reward, value=0, done=False)

        if len(self.rl_agent.states) >= 10:
            self.rl_agent.update_policy()

    def finetune_llm(self):
        descriptions = [f"Valid face with AUs: {face}" for face in self.valid_faces]
        descriptions += [f"Bad face with AUs: {face}" for face in self.bad_faces]

        self.llm.finetune(descriptions)

    def save_models(self):
        os.makedirs("models/rl", exist_ok=True)
        os.makedirs("models/llm", exist_ok=True)

        self.rl_agent.save("models/rl/latest_rl_model.pth")
        self.llm.save("models/llm/latest_llm_model.pth")

        print(f"Models saved after epoch {self.current_epoch}.")
