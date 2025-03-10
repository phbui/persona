import sys
from PyQt6.QtWidgets import QApplication
from ui.trainingwizard import TrainingWizard 

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrainingWizard()
    window.show()
    sys.exit(app.exec())
