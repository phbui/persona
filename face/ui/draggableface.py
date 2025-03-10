from PyQt6.QtWidgets import QListWidgetItem
from PyQt6.QtCore import QMimeData, QByteArray, Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtGui import QDrag

class DraggableFace(QListWidgetItem):
    def __init__(self, au_values, pixmap):
        super().__init__()
        self.au_values = au_values
        self.pixmap = pixmap
        self.setIcon(QIcon(pixmap))
        self.setText("")  

    def mimeData(self):
        data = QMimeData()
        byte_array = QByteArray()
        byte_array.append(str(self.au_values.tolist()))
        data.setData("application/x-qface", byte_array)

        data.setImageData(self.pixmap.toImage())  
        return data

    def startDrag(self):
        drag = QDrag(self.listWidget())
        drag.setMimeData(self.mimeData())
        drag.setPixmap(self.pixmap)  
        drag.exec(Qt.DropAction.MoveAction)
