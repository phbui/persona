from ui.steps.modelselectionstep import ModelSelectionStep
from ui.steps.trainingmodestep import TrainingModeStep
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QStackedWidget
)


class TrainingWizard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Model Selection Wizard")
        self.setGeometry(100, 100, 800, 400)

        self.layout = QVBoxLayout()
        self.stacked_widget = QStackedWidget()
        self.layout.addWidget(self.stacked_widget)
        self.setLayout(self.layout)

        self.rl_model = None
        self.llm = None
        self.training_mode = None  

        self.model_selection_step = ModelSelectionStep(self)
        self.training_mode_step = TrainingModeStep(self)

        self.stacked_widget.addWidget(self.model_selection_step)
        self.stacked_widget.addWidget(self.training_mode_step)

    def show_training_mode_step(self):
        self.stacked_widget.setCurrentWidget(self.training_mode_step)
