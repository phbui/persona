from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit
from PyQt6.QtCore import Qt


class TrainingLogStep(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()
        self.status_label = QLabel("Training Progress")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        layout.addWidget(self.status_label)
        layout.addWidget(self.log_box)
        self.setLayout(layout)

    def append_log(self, text):
        self.log_box.append(text)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def set_status(self, text):
        self.status_label.setText(text)

    def reset(self):
        self.log_box.clear()
        self.status_label.setText("Training Progress")
