import sys
from PyQt6.QtWidgets import QApplication
from trainingui import TrainingUI

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrainingUI()
    window.show()
    sys.exit(app.exec())
