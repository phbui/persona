from PyQt6.QtWidgets import QWidget, QVBoxLayout, QStackedWidget
from ui.training.humanfeedback.epochselectionstep import EpochSelectionStep
from ui.training.humanfeedback.facemarkingstep import FaceMarkingStep
from ui.training.humanfeedback.rankingstep import RankingStep
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from feat.plotting import plot_face
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from io import BytesIO
import json

class HumanFeedbackTrainingWizard(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.epochs = 1
        self.current_epoch = 0
        self.current_situation_index = 0
        self.valid_faces = []
        self.invalid_faces = []
        self.generated_faces = []
        self.situations = self.load_situations()

        layout = QVBoxLayout()
        self.stacked_widget = QStackedWidget()
        layout.addWidget(self.stacked_widget)
        self.setLayout(layout)

        self.epoch_selection_step = EpochSelectionStep(self)
        self.face_marking_step = FaceMarkingStep(self)
        self.ranking_step = RankingStep(self)

        self.stacked_widget.addWidget(self.epoch_selection_step)
        self.stacked_widget.addWidget(self.face_marking_step)
        self.stacked_widget.addWidget(self.ranking_step)

        self.show_epoch_selection_step()

    def load_situations(self):
        try:
            with open("data/situations.json", "r", encoding="utf-8") as file:
                data = json.load(file)
                return data.get("situations", ["Default Situation"])
        except (FileNotFoundError, json.JSONDecodeError):
            return ["Default Situation"]

    def show_epoch_selection_step(self):
        self.stacked_widget.setCurrentWidget(self.epoch_selection_step)

    def start_training(self):
        self.current_epoch = 0
        self.current_situation_index = 0
        self.run_epoch()

    def run_epoch(self):
        self.valid_faces = []
        self.invalid_faces = []
        self.generated_faces = []

        if self.current_epoch >= self.epochs:
            print("Training complete.")
            return

        if self.current_situation_index >= len(self.situations):
            self.current_epoch += 1
            self.current_situation_index = 0
            if self.current_epoch >= self.epochs:
                print("Training complete.")
                return

        self.show_face_marking_step()

    def show_face_marking_step(self):
        self.generated_faces = self.generate_faces()
        situation_text = self.situations[self.current_situation_index]
        self.face_marking_step.display_faces(self.generated_faces, situation_text, self.parent.character_name, self.parent.character_description)
        self.stacked_widget.setCurrentWidget(self.face_marking_step)

    def show_ranking_step(self):
        print(f"Current Valid Faces: {self.valid_faces}")
        print(f"Current Invalid Faces: {self.invalid_faces}")
        if (len(self.valid_faces)) == 0:
            self.ranking_done()
            return
        situation_text = self.situations[self.current_situation_index]
        self.ranking_step.display_faces(self.valid_faces, situation_text, self.parent.character_name, self.parent.character_description)
        self.stacked_widget.setCurrentWidget(self.ranking_step)

    def rank_valid_faces(self):
        self.valid_faces.sort(key=lambda x: self.valid_faces.index(x))
        self.show_ranking_step()

    def ranking_done(self):
        self.submit_human_feedback()
        self.current_situation_index += 1
        self.run_epoch()

    def submit_human_feedback(self):
        print(f"Valid Faces (Ranked): {self.valid_faces}")
        print(f"Invalid Faces: {self.invalid_faces}")

        state = self.parent.manager_extraction.extract_features(self.situations[self.current_situation_index])

        for action_au in self.invalid_faces:
            self.parent.rl_model.store_transition(
                state=state, action=action_au, log_prob=0, reward=-1.0, value=0, done=False
            )

        num_valid = len(self.valid_faces)
        for rank, action_au in enumerate(self.valid_faces):
            reward = 1 - (rank / num_valid) 
            self.parent.rl_model.store_transition(
                state=state, action=action_au, log_prob=0, reward=reward, value=0, done=False
            )

        self.parent.rl_model.update_policy()
        self.parent.rl_model.save_model(self.parent.parent.rl_model_path)
        self.parent.llm_model.save_model(self.parent.parent.llm_model_path)


    def generate_faces(self):
        state = self.parent.manager_extraction.extract_features(self.situations[self.current_situation_index])
        state_tensor = th.tensor(state, dtype=th.float32).unsqueeze(0)
        state_tensor = state_tensor.to(next(self.parent.rl_model.policy.parameters()).device)

        faces = []
        for _ in range(10):
            action, _, _ = self.parent.rl_model.policy.select_action(state_tensor)
            action_au = np.clip(action, 0, 3)
            faces.append(action_au)

        return faces

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
