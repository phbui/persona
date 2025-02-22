import json
import os
from meta.meta_singleton import Meta_Singleton
from log.logger import Logger


class Manager_File(metaclass=Meta_Singleton):
    """
    Manager_File class for managing JSON file upload and download operations.
    This class is implemented as a singleton, ensuring only one instance exists.
    """
    def __init__(self):
        self.logger = Logger()

    def upload_file(self, file_path):
        """
        Reads JSON data from the specified file path.

        :param file_path: str - Path to the JSON file to be read.
        :return: dict or list containing JSON data if successful; otherwise, None.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            self.logger.add_log("INFO", "uploads", self.__class__.__name__, "upload_file", f"Data successfully uploaded from {file_path}")
            return data
        except Exception as e:
            self.logger.add_log("ERROR", "uploads", self.__class__.__name__, "upload_file", f"Error reading JSON from {file_path}: {e}")
            return None

    def download_file(self, data, dir_path, filename="downloaded.json"):
        """
        Writes the provided JSON data to a file in the specified directory.

        If the directory does not exist, it will be created.

        :param data: dict or list - JSON data to be written to file.
        :param dir_path: str - Directory path where the file should be saved.
        :param filename: str - Name of the file to be saved (default "downloaded.json").
        :return: True if file was written successfully; otherwise, False.
        """
        # Ensure the directory exists; if not, create it.
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)
                self.logger.add_log("INFO", "downloads", self.__class__.__name__, "download_file", f"Directory {dir_path} created.")
            except Exception as e:
                self.logger.add_log("ERROR", "downloads", self.__class__.__name__, "download_file", f"Error creating directory {dir_path}: {e}")
                return False

        file_path = os.path.join(dir_path, filename)
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
            self.logger.add_log("INFO", "downloads", self.__class__.__name__, "download_file", f"Data successfully downloaded to {file_path}")
            return True
        except Exception as e:
            self.logger.add_log("ERROR", "downloads", self.__class__.__name__, "download_file", f"Error writing JSON to {file_path}: {e}")
            return False