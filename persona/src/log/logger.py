import datetime
import json
from threading import Lock
from ..meta.meta_singleton import Meta_Singleton

class Logger(metaclass=Meta_Singleton):
    """
    A Singleton Logger class that manages log creation and storage.
    """

    LOG_LEVELS = {
        "DEBUG": 1,
        "INFO": 2,
        "WARNING": 3,
        "ERROR": 4,
        "CRITICAL": 5
    }

    def __init__(self, log_level="INFO"):
        self.log_level = self.LOG_LEVELS.get(log_level.upper(), 2)
        self.logs = []
        self.lock = Lock()

    def add_log(self, log_type, data_type, cls_name, func_name, details):
        """
        Adds a log entry with a timestamp.

        :param log_type: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        :param cls_name: Name of the class where the log originates
        :param func_name: Name of the function where the log originates
        :param details: Additional log details
        :param data_type: A string representing the type of data being logged
        """
        if self.LOG_LEVELS.get(log_type.upper(), 0) < self.log_level:
            return

        timestamp = datetime.datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "type": log_type.upper(),
            "data_type": data_type,
            "class": cls_name,
            "function": func_name,
            "details": details,
        }

        with self.lock:
            self.logs.append(log_entry)

        print(json.dumps(log_entry, indent=4))

    def get_logs(self):
        """Returns all logs."""
        return self.logs

    def get_filtered_logs(self, start_time=None, end_time=None, log_type=None, data_type=None,  cls_name=None, func_name=None):
        with self.lock:
            filtered_logs = self.logs

            if start_time:
                start_time = datetime.datetime.fromisoformat(start_time)
                filtered_logs = [log for log in filtered_logs if datetime.datetime.fromisoformat(log["timestamp"]) >= start_time]

            if end_time:
                end_time = datetime.datetime.fromisoformat(end_time)
                filtered_logs = [log for log in filtered_logs if datetime.datetime.fromisoformat(log["timestamp"]) <= end_time]

            if log_type:
                filtered_logs = [log for log in filtered_logs if log["type"] == log_type.upper()]

            if data_type:
                filtered_logs = [log for log in filtered_logs if log["data_type"] == data_type]

            if cls_name:
                filtered_logs = [log for log in filtered_logs if log["class"] == cls_name]

            if func_name:
                filtered_logs = [log for log in filtered_logs if log["function"] == func_name]

        return filtered_logs