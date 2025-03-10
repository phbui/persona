from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget
from PyQt6.QtGui import QPixmap, QDragEnterEvent, QDropEvent
from PyQt6.QtCore import Qt

class DropZone(QWidget):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank
        self.layout = QVBoxLayout()
        self.rank_label = QLabel(f"Rank {rank}")
        self.rank_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.rank_label)
        self.face_label = QLabel()
        self.layout.addWidget(self.face_label)
        self.setLayout(self.layout)
        self.face = None  

        self.setAcceptDrops(True)  

    def set_face(self, face_item):
        self.face_label.setPixmap(face_item.pixmap.scaled(150, 150, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.face_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.face = face_item.au_values  

    def clear(self):
        self.face_label.clear()
        self.face = None  

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasImage():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasImage():
            image = event.mimeData().imageData()
            pixmap = QPixmap.fromImage(image)
            self.face_label.setPixmap(pixmap.scaled(150, 150, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

            self.face = event.mimeData().text()  
            event.acceptProposedAction()
