
from PyQt6.QtWidgets import ( QListWidgetItem, 
)
from PyQt6.QtCore import QMimeData, QByteArray
from PyQt6.QtGui import QIcon

class DraggableFace(QListWidgetItem):
    def __init__(self, au_values, pixmap):
        super().__init__()
        self.au_values = au_values
        self.setIcon(QIcon(pixmap))
        self.setText("")  # Remove face name

    def mimeData(self):
        data = QMimeData()
        byte_array = QByteArray()
        byte_array.append(str(self.au_values.tolist()))
        data.setData("application/x-qface", byte_array)
        return data

