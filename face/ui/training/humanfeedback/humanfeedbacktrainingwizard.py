from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QStackedWidget
)
from ui.training.humanfeedback.epochselectionstep import EpochSelectionStep
from ui.training.humanfeedback.facemarkingstep import FaceMarkingStep
from ui.training.humanfeedback.rankingstep import RankingStep

class HumanFeedbackTrainingWizard(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.epochs = 1
        self.valid_faces = []
        self.invalid_faces = []

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

    def show_epoch_selection_step(self):
        self.stacked_widget.setCurrentWidget(self.epoch_selection_step)

    def show_face_marking_step(self):
        self.stacked_widget.setCurrentWidget(self.face_marking_step)

    def show_ranking_step(self):
        self.stacked_widget.setCurrentWidget(self.ranking_step)
        self.ranking_step.enable_buttons()

    def rank_valid_faces(self):
        print("Ranking valid faces:", self.valid_faces)

    def submit_ranking(self):
        print("Submitting ranking...")
        print("Valid Faces:", self.valid_faces)
        print("Invalid Faces:", self.invalid_faces)
