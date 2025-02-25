import os
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QFileDialog
from manager.manager_file import Manager_File
from log.logger import Logger, Log

class Interface(QWidget):

    def __init__(self, title: str, description: str):
        super().__init__()
        self.title = title
        self.description = description
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()

        title_label = QLabel(self.title)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title_label)

        desc_label = QLabel(self.description)
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        self.setLayout(layout)

    def upload_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open JSON File",
            "",
            "JSON Files (*.json)"
        )
        if not file_path:
            return None

        manager = Manager_File()
        data = manager.upload_file(file_path)

        # Log the upload operation
        logger = Logger()
        log = Log("INFO", "uploads", self.__class__.__name__, "upload_file", f"File {file_path} processed via interface.")
        logger.add_log(log)

        return data

    def download_file(self, data):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save JSON File",
            "",
            "JSON Files (*.json)"
        )
        if not file_path:
            return False

        dir_path = os.path.dirname(file_path)
        filename = os.path.basename(file_path)

        manager = Manager_File()
        success = manager.download_file(data, dir_path, filename)

        logger = Logger()
        if success:
            log = Log("INFO", "downloads", self.__class__.__name__, "download_file", f"File {filename} successfully downloaded to {dir_path} via interface.")
        else:
            log = Log("ERROR", "downloads", self.__class__.__name__, "download_file", f"Failed to download file {filename} to {dir_path} via interface.")
        logger.add_log(log)

        return success
