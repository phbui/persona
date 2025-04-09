from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QStackedWidget
)
from PyQt6.QtCore import Qt, QTimer
import numpy as np
import torch as th

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from ui.training.epochselectionstep import EpochSelectionStep
from ui.training.autotraining.traininglogstep import TrainingLogStep 
import json
import os

class AutoTrainingWizard(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.epochs = 1
        self.current_epoch = 0
        self.current_situation_index = 0
        self.llm_training = []
        self.situations = self.load_situations()

        layout = QVBoxLayout()
        self.stacked_widget = QStackedWidget()
        layout.addWidget(self.stacked_widget)
        self.setLayout(layout)

        self.epoch_selection_step = EpochSelectionStep(self)
        self.training_log_step = TrainingLogStep(self)  

        self.reward_plot_page = QWidget()
        reward_layout = QVBoxLayout()
        self.reward_info_label = QLabel("Average Reward per Epoch")
        self.reward_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        reward_layout.addWidget(self.reward_info_label)

        self.reward_fig = Figure(figsize=(5, 4))
        self.reward_ax = self.reward_fig.add_subplot(111)
        self.canvas = FigureCanvas(self.reward_fig)
        reward_layout.addWidget(self.canvas)
        self.reward_plot_page.setLayout(reward_layout)

        self.stacked_widget.addWidget(self.epoch_selection_step)
        self.stacked_widget.addWidget(self.training_log_step)
        self.stacked_widget.addWidget(self.reward_plot_page)

        self.show_epoch_selection_step()

    def load_situations(self):
        file_path = "data/situations_expanded.json"
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

    def post_epoch_step(self):
        self.current_epoch = 0
        self.current_situation_index = 0
        self.llm_training = []
        self.training_log_step.reset()
        self.training_log_step.append_log("Starting training...\n")
        self.stacked_widget.setCurrentWidget(self.training_log_step)
        QTimer.singleShot(100, self.run_epoch)

    def run_epoch(self):
        if self.current_epoch > self.epochs:
            self.show_epoch_plot()
            return

        if self.current_situation_index >= len(self.situations):
            self.parent.manager_reward.end_epoch(self.current_epoch)
            self.parent.llm_model.fine_tune(self.llm_training, self.parent.llm_model_path)
            self.llm_training = []
            self.training_log_step.append_log(f"\nEpoch {self.current_epoch + 1} complete.\n")
            self.current_epoch += 1
            self.current_situation_index = 0
            QTimer.singleShot(100, self.run_epoch)
            return

        situation_text = self.situations[self.current_situation_index]
        self.training_log_step.set_status(
            f"Epoch {self.current_epoch+1} | Episode {self.current_situation_index+1}:\n{situation_text}"
        )

        generated_faces = self.generate_faces()

        valid_faces, invalid_faces, response = self.parent.llm_model.auto_generate_face_feedback(
            self.parent.character_description,
            situation_text,
            generated_faces,
            self.parent.manager_extraction.describe_face
        )

        state = self.parent.manager_extraction.extract_features(situation_text, True)

        for face in invalid_faces:
            self.parent.rl_model.store_transition(
                state=state, action=face, log_prob=0, reward=-2.0, value=0, done=False
            )
            self.parent.manager_reward.store_reward(-2.0)

        num_valid = len(valid_faces)
        for rank, face in enumerate(valid_faces):
            reward = 1 - (rank / num_valid) if num_valid > 0 else 0
            self.parent.rl_model.store_transition(
                state=state, action=face, log_prob=0, reward=reward, value=0, done=False
            )
            self.parent.manager_reward.store_reward(reward)

        self.parent.manager_reward.end_episode()

        episode_reward = self.parent.manager_reward.epoch_rewards[-1]
        self.training_log_step.append_log(
            f"[Epoch {self.current_epoch + 1}, Episode {self.current_situation_index + 1} | Situation: {self.situations[self.current_situation_index]}, Valid: {num_valid}] {episode_reward:.2f}"
        )

        self.parent.rl_model.update_policy(self.parent.rl_model_path)
        self.current_situation_index += 1
        QTimer.singleShot(100, self.run_epoch)

    def show_epoch_plot(self):
        epochs, rewards = self.parent.manager_reward.load_rewards()
        self.reward_ax.clear()
        self.reward_ax.plot(epochs, rewards, marker='o')
        self.reward_ax.set_title("Average Reward per Epoch")
        self.reward_ax.set_xlabel("Epoch")
        self.reward_ax.set_ylabel("Avg Reward")
        self.canvas.draw()
        self.stacked_widget.setCurrentWidget(self.reward_plot_page)

    def generate_faces(self):
        situation_text = self.situations[self.current_situation_index]
        state = self.parent.manager_extraction.extract_features(situation_text, True)
        state_tensor = th.tensor(state, dtype=th.float32).unsqueeze(0)
        state_tensor = state_tensor.to(next(self.parent.rl_model.policy.parameters()).device)

        faces = []
        for _ in range(10):
            action, _, _ = self.parent.rl_model.policy.select_action(state_tensor)
            action_au = np.clip(action, 0, 3)
            faces.append(action_au)
        return faces
