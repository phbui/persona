import os
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QFileDialog
from PyQt6.QtCore import Qt
from manager.manager_file import Manager_File
from log.logger import Logger, Log

class Interface(QWidget):
    """
    A base class for all tab interfaces that provides a common look & feel.
    Each derived class can supply its own title and description (or override any methods as needed).
    """

    def __init__(self, title: str, description: str):
        super().__init__()
        self.title = title
        self.description = description
        self._init_ui()

    def _init_ui(self):
        """
        Create a common layout and styling for all derived interfaces.
        """
        layout = QVBoxLayout()

        title_label = QLabel(self.title)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title_label)

        desc_label = QLabel(self.description)
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        self.setLayout(layout)

    def upload_file(self):
        """
        Opens a file explorer window to select a JSON file for uploading.
        Uses the file manager to process the selected file and logs the operation.
        :return: The uploaded data, or None if no file was selected or an error occurred.
        """
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
        """
        Opens a file explorer window to choose where to save the JSON file.
        Uses the file manager to download the data and logs the operation.
        :param data: JSON data to write.
        :return: True if the file was successfully written; otherwise, False.
        """
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save JSON File",
            "",
            "JSON Files (*.json)"
        )
        if not file_path:
            return False

        # Split the file path into directory and filename.
        dir_path = os.path.dirname(file_path)
        filename = os.path.basename(file_path)

        manager = Manager_File()
        success = manager.download_file(data, dir_path, filename)

        # Log the download operation based on success or failure.
        logger = Logger()
        if success:
            log = Log("INFO", "downloads", self.__class__.__name__, "download_file", f"File {filename} successfully downloaded to {dir_path} via interface.")
        else:
            log = Log("ERROR", "downloads", self.__class__.__name__, "download_file", f"Failed to download file {filename} to {dir_path} via interface.")
        logger.add_log(log)

        return success
