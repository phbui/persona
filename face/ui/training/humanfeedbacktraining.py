import os
import json
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from feat.plotting import plot_face
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from ai.manager_extraction import Manager_Extraction
from ui.training.humanfeedbacktraining_epochselection import EpochSelectionUI
from ui.training.humanfeedbacktraining_faceselection import FaceSelectionUI
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

        self.epoch_selection_ui = EpochSelectionUI(self)
        self.face_selection_ui = FaceSelectionUI(self)

        self.start_training_button = QPushButton("Start Training")
        self.start_training_button.setEnabled(False)
        self.start_training_button.clicked.connect(self.start_training)

        layout.addWidget(self.epoch_selection_ui)
        layout.addWidget(self.start_training_button)
        layout.addWidget(self.face_selection_ui)

        self.setLayout(layout)

        self.current_epoch = 0
        self.current_situation_index = 0
        self.generated_faces = []
        self.valid_faces = []
        self.bad_faces = []
        self.checkboxes = []

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
        self.face_selection_ui.situation_label.setText(f"Situation: {situation}")

        state = self.manager_extraction.extract_features(situation)
        self.generated_faces = self.generate_faces(state)

        self.face_selection_ui.display_faces(self.generated_faces)
        self.valid_faces = []
        self.bad_faces = []
        self.checkboxes = []

        self.face_selection_ui.rank_faces_button.setEnabled(False)
        self.face_selection_ui.submit_button.setEnabled(False)

    def generate_faces(self, state):
        state_tensor = th.tensor(state, dtype=th.float32).unsqueeze(0)
        state_tensor = state_tensor.to(next(self.rl_agent.policy.parameters()).device)

        faces = []
        for _ in range(10):
            action, _, _ = self.rl_agent.policy.select_action(state_tensor)
            action_au = np.clip(action, 0, 3)
            faces.append(action_au)

        return faces
    
    def rank_valid_faces(self):
        self.valid_faces.sort(key=lambda x: self.valid_faces.index(x))
        self.face_selection_ui.submit_button.setEnabled(True)

    def generate_face_pixmap(self, au_values, size=(200, 200)):
        fig, ax = plt.subplots(figsize=(8, 9), dpi=400)
        plot_face(ax=ax, au=au_values)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=400, facecolor="white")
        buf.seek(0)

        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue(), "PNG")
        buf.close()

        return pixmap.scaled(size[0], size[1], Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

    def toggle_valid_bad(self, checked_box, other_box, au_values):
        if checked_box.isChecked():
            other_box.setChecked(False)
        self.update_valid_bad_faces()

    def update_valid_bad_faces(self):
        self.valid_faces = [au for v, b, au in self.checkboxes if v.isChecked()]
        self.bad_faces = [au for v, b, au in self.checkboxes if b.isChecked()]
        self.face_selection_ui.rank_faces_button.setEnabled(bool(self.valid_faces))
        self.face_selection_ui.submit_button.setEnabled(True)

    def submit_ranking(self):
        self.train_rl_model()
        self.finetune_llm()
        self.current_situation_index += 1
        self.process_next_situation()
