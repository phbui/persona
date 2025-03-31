from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from feat.plotting import plot_face
from io import BytesIO
import json

class AutoTrainingWizard(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        
        # Configuration: total number of epochs to run
        self.epochs = 1
        self.current_epoch = 0
        self.current_situation_index = 0
        
        # Load situations from file.
        self.situations = self.load_situations()
        
        # Build a simple UI for status reporting.
        layout = QVBoxLayout()
        self.info_label = QLabel("Auto Training Mode: Click 'Start Training' to begin")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info_label)
        
        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self.start_training)
        layout.addWidget(self.start_button)
        
        self.setLayout(layout)
    
    def load_situations(self):
        try:
            with open("data/situations.json", "r", encoding="utf-8") as file:
                data = json.load(file)
                return data.get("situations", ["Default Situation"])
        except (FileNotFoundError, json.JSONDecodeError):
            return ["Default Situation"]
    
    def start_training(self):
        self.current_epoch = 0
        self.current_situation_index = 0
        self.info_label.setText("Training started...")
        self.start_button.setEnabled(False)
        self.run_epoch()
    
    def run_epoch(self):
        # Check if all epochs are completed.
        if self.current_epoch >= self.epochs:
            self.info_label.setText("Training complete.")
            return
        
        # End of the current epoch: if we've processed all situations.
        if self.current_situation_index >= len(self.situations):
            # Save rewards and fine-tune LLM as in human feedback.
            self.parent.manager_reward.end_epoch(self.current_epoch)
            self.parent.llm_model.fine_tune(self.llm_training, self.parent.llm_model_path)
            self.llm_training = []
            
            self.current_epoch += 1
            self.current_situation_index = 0
            
            if self.current_epoch >= self.epochs:
                self.info_label.setText("Training complete.")
                return
        
        # Process the current situation (episode).
        situation_text = self.situations[self.current_situation_index]
        self.info_label.setText(f"Epoch {self.current_epoch+1}, Situation {self.current_situation_index+1}:\n{situation_text}")
        
        # Generate candidate faces using the RL policy.
        generated_faces = self.generate_faces()
        
        # Use the fine-tuned LLM to automatically generate feedback.
        # This call uses the parent's face description function, e.g. manager_extraction.describe_face.
        valid_faces, invalid_faces, response = self.parent.llm_model.auto_generate_face_feedback(
            self.parent.character_description,
            situation_text,
            generated_faces,
            self.parent.manager_extraction.describe_face
        )
        
        # Extract the state for the current situation.
        state = self.parent.manager_extraction.extract_features(situation_text)
        print(f"Response: {response}")
        
        # Record transitions for invalid faces (with a negative reward).
        for face in invalid_faces:
            self.parent.rl_model.store_transition(
                state=state, action=face, log_prob=0, reward=-1.0, value=0, done=False
            )
            self.parent.manager_reward.store_reward(-1.0)
        
        # Record transitions for valid faces with rank-based rewards.
        num_valid = len(valid_faces)
        for rank, face in enumerate(valid_faces):
            reward = 1 - (rank / num_valid) if num_valid > 0 else 0
            self.parent.rl_model.store_transition(
                state=state, action=face, log_prob=0, reward=reward, value=0, done=False
            )
            self.parent.manager_reward.store_reward(reward)
        
        # Mark the end of the episode.
        self.parent.manager_reward.end_episode()
        
        # Update the RL policy.
        self.parent.rl_model.update_policy(self.parent.rl_model_path)
        # Move to the next situation.
        self.current_situation_index += 1
        
        # Continue processing the next episode.
        self.run_epoch()
    
    def generate_faces(self):
        situation_text = self.situations[self.current_situation_index]
        state = self.parent.manager_extraction.extract_features(situation_text)
        state_tensor = th.tensor(state, dtype=th.float32).unsqueeze(0)
        state_tensor = state_tensor.to(next(self.parent.rl_model.policy.parameters()).device)
        
        faces = []
        for _ in range(10):
            action, _, _ = self.parent.rl_model.policy.select_action(state_tensor)
            action_au = np.clip(action, 0, 3)  # clip action values as in human feedback
            faces.append(action_au)
        return faces
    