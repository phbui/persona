from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QStackedWidget
)
from PyQt6.QtCore import Qt
import numpy as np
import torch as th
import json

# For plotting
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# Import your epoch selection widget.
from ui.training.epochselectionstep import EpochSelectionStep

class AutoTrainingWizard(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        
        # Default configuration values.
        self.epochs = 1
        self.current_epoch = 0
        self.current_situation_index = 0
        self.llm_training = []
        self.situations = self.load_situations()
        
        # Setup a stacked widget to handle multiple UI pages.
        layout = QVBoxLayout()
        self.stacked_widget = QStackedWidget()
        layout.addWidget(self.stacked_widget)
        self.setLayout(layout)
        
        # Create the epoch selection step.
        self.epoch_selection_step = EpochSelectionStep(self)
        
        # Create the auto training page (status reporting during training).
        self.auto_training_page = QWidget()
        auto_training_layout = QVBoxLayout()
        self.info_label = QLabel("Auto Training Mode")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        auto_training_layout.addWidget(self.info_label)
        self.auto_training_page.setLayout(auto_training_layout)
        
        # Create the reward plot page.
        self.reward_plot_page = QWidget()
        reward_layout = QVBoxLayout()
        self.reward_info_label = QLabel("Reward per Episode")
        self.reward_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        reward_layout.addWidget(self.reward_info_label)
        
        # Create a matplotlib figure and canvas.
        self.reward_fig = Figure(figsize=(5, 4))
        self.reward_ax = self.reward_fig.add_subplot(111)
        self.canvas = FigureCanvas(self.reward_fig)
        reward_layout.addWidget(self.canvas)
        
        # Add a continue button to resume training.
        self.continue_button = QPushButton("Continue Training")
        self.continue_button.clicked.connect(self.resume_training)
        reward_layout.addWidget(self.continue_button)
        self.reward_plot_page.setLayout(reward_layout)
        
        # Add pages to the stacked widget.
        self.stacked_widget.addWidget(self.epoch_selection_step)
        self.stacked_widget.addWidget(self.auto_training_page)
        self.stacked_widget.addWidget(self.reward_plot_page)
        
        # Start by showing the epoch selection step.
        self.show_epoch_selection_step()
    
    def show_epoch_selection_step(self):
        self.stacked_widget.setCurrentWidget(self.epoch_selection_step)
    
    def load_situations(self):
        try:
            with open("data/situations.json", "r", encoding="utf-8") as file:
                data = json.load(file)
                return data.get("situations", ["Default Situation"])
        except (FileNotFoundError, json.JSONDecodeError):
            return ["Default Situation"]
        
    def post_epoch_step(self):
        self.start_training()
    
    def start_training(self):
        # Retrieve the selected number of epochs from the epoch selection step.
        # (Assuming the epoch selection widget stores the chosen number in an attribute `selected_epochs`.)
        self.epochs = getattr(self.epoch_selection_step, "selected_epochs", 1)
        self.current_epoch = 0
        self.current_situation_index = 0
        self.llm_training = []
        
        # Switch to the auto training UI.
        self.stacked_widget.setCurrentWidget(self.auto_training_page)
        self.info_label.setText("Training started...")
        
        # Begin processing episodes.
        self.run_epoch()
    
    def run_epoch(self):
        # Check if all epochs are completed.
        if self.current_epoch >= self.epochs:
            self.info_label.setText("Training complete.")
            return
        
        # If we have processed all situations in the current epoch,
        # finalize the epoch: end rewards, fine-tune the LLM, and reset for the next epoch.
        if self.current_situation_index >= len(self.situations):
            self.parent.manager_reward.end_epoch(self.current_epoch)
            self.parent.llm_model.fine_tune(self.llm_training, self.parent.llm_model_path)
            self.llm_training = []
            
            self.current_epoch += 1
            self.current_situation_index = 0
            
            # If training is finished, update the UI accordingly.
            if self.current_epoch >= self.epochs:
                self.info_label.setText("Training complete.")
                return
            # After an epoch, show the reward plot before continuing.
            self.show_reward_plot()
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
        
        # Debug print the LLM response.
        print(f"LLM Response: {response}")
        
        # Extract state features for the current situation.
        state = self.parent.manager_extraction.extract_features(situation_text)
        
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
        
        # After processing an episode, update the reward plot and switch to that UI.
        self.show_reward_plot()
    
    def show_reward_plot(self):
        """
        Updates the reward plot using rewards recorded by manager_reward.
        Assumes manager_reward.get_episode_rewards() returns a list of rewards per episode.
        """
        # Retrieve the list of episode rewards.
        rewards = self.parent.manager_reward.get_episode_rewards()
        
        # Clear the axes and plot the updated reward data.
        self.reward_ax.clear()
        self.reward_ax.plot(rewards, marker='o')
        self.reward_ax.set_title("Reward per Episode")
        self.reward_ax.set_xlabel("Episode")
        self.reward_ax.set_ylabel("Reward")
        self.canvas.draw()
        
        # Switch to the reward plot UI.
        self.stacked_widget.setCurrentWidget(self.reward_plot_page)
    
    def resume_training(self):
        """
        Called when the user clicks the 'Continue Training' button.
        Switches back to the auto training page and processes the next episode.
        """
        self.stacked_widget.setCurrentWidget(self.auto_training_page)
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
