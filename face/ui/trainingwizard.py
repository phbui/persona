from ui.steps.characterdescriptionstep import CharacterDescriptionStep
from ui.steps.modelselectionstep import ModelSelectionStep
from ui.steps.trainingmodestep import TrainingModeStep
from ui.training.humanfeedback.humanfeedbacktrainingwizard import HumanFeedbackTrainingWizard
from ui.training.autotraining.autotrainingwizard import AutoTrainingWizard
from ui.training.manager_rewards import Manager_Reward
from ai.manager_extraction import Manager_Extraction
from ui.novelgenerationstep import NovelGenerationStep

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QStackedWidget
)

class TrainingWizard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Training Wizard")
        self.setGeometry(10, 10, 800, 1000)

        self.layout = QVBoxLayout()
        self.stacked_widget = QStackedWidget()
        self.layout.addWidget(self.stacked_widget)
        self.setLayout(self.layout)

        self.rl_model = None
        self.llm_model = None
        self.training_mode = None  

        self.manager_extraction = Manager_Extraction()
        self.character_description_step = CharacterDescriptionStep(self)
        self.model_selection_step = ModelSelectionStep(self)
        self.training_mode_step = TrainingModeStep(self)
        self.human_feedback_training = HumanFeedbackTrainingWizard(self)
        self.auto_training = AutoTrainingWizard(self)
        self.novel_generation_step = NovelGenerationStep(self)
        self.manager_reward = Manager_Reward()

        self.stacked_widget.addWidget(self.character_description_step)
        self.stacked_widget.addWidget(self.model_selection_step)
        self.stacked_widget.addWidget(self.training_mode_step)
        self.stacked_widget.addWidget(self.human_feedback_training)
        self.stacked_widget.addWidget(self.auto_training)
        self.stacked_widget.addWidget(self.novel_generation_step)

    def show_model_selection_step(self):
        self.stacked_widget.setCurrentWidget(self.model_selection_step)

    def show_training_mode_step(self):
        self.stacked_widget.setCurrentWidget(self.training_mode_step)

    def show_novel_generation_step(self):
        self.stacked_widget.setCurrentWidget(self.novel_generation_step)
        self.rl_model.policy.set_exploit()
        self.novel_generation_step.run_generation()

    def show_training_mode_step(self):
        if not self.rl_model or not self.llm_model:
            print("Error: Models must be selected before proceeding.")
            return
        self.stacked_widget.setCurrentWidget(self.training_mode_step)

    def show_training_step(self):
        if self.training_mode == "human_feedback":
            self.training_widget = HumanFeedbackTrainingWizard(self)
        elif self.training_mode == "auto_training":
            self.training_widget = AutoTrainingWizard(self)
            self.rl_model.policy.set_auto()
        elif self.training_mode == "novel_generation":
            self.show_novel_generation_step()
            return

        self.stacked_widget.addWidget(self.training_widget)
        self.stacked_widget.setCurrentWidget(self.training_widget)
