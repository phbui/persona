import json
import os
from meta.meta_singleton import Meta_Singleton
from log.logger import Logger


class Manager_File(metaclass=Meta_Singleton):
    def __init__(self):
        self.logger = Logger()

    def upload_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            self.logger.add_log("INFO", "uploads", self.__class__.__name__, "upload_file", f"Data successfully uploaded from {file_path}")
            return data
        except Exception as e:
            self.logger.add_log("ERROR", "uploads", self.__class__.__name__, "upload_file", f"Error reading JSON from {file_path}: {e}")
            return None

    def download_file(self, data, file_path, file_name="downloaded.json"):
        if not os.path.exists(file_path):
            try:
                os.makedirs(file_path)
                self.logger.add_log("INFO", "downloads", self.__class__.__name__, "download_file", f"Directory {file_path} created.")
            except Exception as e:
                self.logger.add_log("ERROR", "downloads", self.__class__.__name__, "download_file", f"Error creating directory {file_path}: {e}")
                return False

        file_path = os.path.join(file_path, file_name)
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
            self.logger.add_log("INFO", "downloads", self.__class__.__name__, "download_file", f"Data successfully downloaded to {file_path}")
            return True
        except Exception as e:
            self.logger.add_log("ERROR", "downloads", self.__class__.__name__, "download_file", f"Error writing JSON to {file_path}: {e}")
            return False