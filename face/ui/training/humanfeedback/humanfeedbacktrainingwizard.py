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
        self.current_epoch = 0  
        self.valid_faces = []
        self.invalid_faces = []
        self.generated_faces = []

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

    def start_training(self):
        self.current_epoch = 0  
        self.run_epoch()

    def run_epoch(self):
        if self.current_epoch >= self.epochs:
            print("✅ Training complete.")
            return

        print(f"🚀 Starting epoch {self.current_epoch + 1} / {self.epochs}")
        self.show_face_marking_step()

    def show_face_marking_step(self):
        self.generated_faces = self.parent.generate_faces()
        self.face_marking_step.display_faces(self.generated_faces)
        self.stacked_widget.setCurrentWidget(self.face_marking_step)

    def show_ranking_step(self):
        if self.valid_faces:
            self.stacked_widget.setCurrentWidget(self.ranking_step)
            self.ranking_step.enable_buttons()

    def rank_valid_faces(self):
        self.valid_faces.sort(key=lambda x: self.valid_faces.index(x))
        self.show_ranking_step()

    def ranking_done(self):
        self.parent.submit_ranking()
        self.current_epoch += 1
        self.run_epoch()
