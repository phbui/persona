from PyQt6.QtWidgets import QWidget, QVBoxLayout, QStackedWidget
from ui.training.epochselectionstep import EpochSelectionStep
from ui.training.humanfeedback.facemarkingstep import FaceMarkingStep
from ui.training.humanfeedback.rankingstep import RankingStep
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from feat.plotting import plot_face
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from io import BytesIO
import sys
import json
import os

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
        self.llm_training = []
        self.last_situation_embedding = []

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
        file_path = "data/situations.json"
        print(f"Loading situations from {file_path}.")
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    return data.get("situations", ["Default Situation"])
            except (json.JSONDecodeError, KeyError):
                print("Failed to load situations")
                return ["Default Situation"]
        print("No situations found")
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
            sys.exit(0)
            return

        if self.current_situation_index >= len(self.situations):
            print(f"Finised epoch {self.current_epoch}.")
            self.parent.manager_reward.end_epoch(self.current_epoch)
            loss = self.parent.llm_model.fine_tune(self.llm_training, self.parent.llm_model_path)
            self.parent.manager_loss.store_loss(self.current_epoch, loss)
            self.llm_training = []
            self.current_epoch += 1
            self.current_situation_index = 0
            if self.current_epoch >= self.epochs:
                print("Training complete.")
                sys.exit(0)
                return

        self.post_epoch_step()

    def post_epoch_step(self):
        self.generated_faces = self.generate_faces()
        situation_text = self.situations[self.current_situation_index]
        self.face_marking_step.display_faces(self.generated_faces, situation_text, self.parent.character_name, self.parent.character_description)
        self.stacked_widget.setCurrentWidget(self.face_marking_step)

    def go_back_to_face_marking_step(self):
        situation_text = self.situations[self.current_situation_index]
        self.face_marking_step.display_faces(self.generated_faces, situation_text, self.parent.character_name, self.parent.character_description)
        self.stacked_widget.setCurrentWidget(self.face_marking_step)

    def show_ranking_step(self):
        plt.close('all')
        if (len(self.valid_faces)) <= 1:
            self.ranking_done()
            return
        situation_text = self.situations[self.current_situation_index]
        self.ranking_step.display_faces(self.valid_faces, situation_text, self.parent.character_name, self.parent.character_description)
        self.stacked_widget.setCurrentWidget(self.ranking_step)

    def rank_valid_faces(self):
        self.valid_faces.sort(key=lambda x: self.valid_faces.index(x))
        self.show_ranking_step()

    def ranking_done(self):
        plt.close('all')
        self.submit_human_feedback()
        self.current_situation_index += 1
        self.run_epoch()

    def submit_human_feedback(self):
        state = self.last_situation_embedding

        for face in self.invalid_faces:
            reward = -0.2
            self.parent.rl_model.store_transition(
                state=state, action=face['aus'], log_prob=face['log_prob'], reward=reward, value=face['value'], done=False
            )
            self.parent.manager_reward.store_reward(reward)

        num_valid = len(self.valid_faces)
        #(f"Valid: {self.valid_faces}")
        #(f"Invalid: {self.invalid_faces}")
        for rank, face in enumerate(self.valid_faces):
            reward = (1 - (rank / num_valid)) * 3
            self.parent.rl_model.store_transition(
                state=state, action=face['aus'], log_prob=face['log_prob'], reward=reward, value=face['value'], done=False
            )
            self.parent.manager_reward.store_reward(reward)
        
        self.parent.manager_reward.end_episode()
        self.parent.rl_model.update_policy(self.parent.rl_model_path)

        face_descriptions = "Generated Faces:\n"

        faces = self.generated_faces
        for i, face in enumerate(faces):
            face_descriptions += f"{i}: " + self.parent.manager_extraction.describe_face(face['aus']) + "\n"
            valid_faces_idx = [
                np.where([
                    np.array_equal(face['aus'], gen_face['aus']) for gen_face in self.generated_faces
                ])[0][0]
                for face in self.valid_faces
            ]
            invalid_faces_idx = [
                np.where([
                    np.array_equal(face['aus'], gen_face['aus']) for gen_face in self.generated_faces
                ])[0][0]
                for face in self.invalid_faces
            ]
        response, prompt = self.parent.llm_model.generate_training_text(self.parent.character_description, self.situations[self.current_situation_index], face_descriptions, valid_faces_idx, invalid_faces_idx)
        training_data = {
            "prompt": prompt.replace("\n", " ").strip(),
            "response": response.replace("\n", " ").strip()
        }
        self.llm_training.append(training_data)

    def generate_faces(self):
        state = self.parent.manager_extraction.extract_features(
            self.situations[self.current_situation_index]
        )
        self.last_situation_embedding = state

        faces = []
        for _ in range(5):
            noisy_state = state + np.random.normal(0, 0.5, state.shape)
            state_tensor = th.tensor(noisy_state, dtype=th.float32).unsqueeze(0)
            dynamic_temp = max(2.0 - 0.3 * self.current_epoch, 0.1)
            action, log_prob, value = self.parent.rl_model.policy.select_action(
                state_tensor, dynamic_temp
            )
            
            faces.append({'aus': np.clip(action, 0, 3), 
                        'log_prob': log_prob, 
                        'value': value})
        
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
