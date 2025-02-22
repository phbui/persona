import datetime
from threading import Lock
from meta.meta_singleton import Meta_Singleton
from .log import Log

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

    def __init__(self, log_level="WARNING"):
        self.log_level = self.LOG_LEVELS.get(log_level.upper(), 2)
        self.logs = []
        self.lock = Lock()

    def add_log_obj(self, log: Log):
        """
        Adds a Log object entry if it meets the required log level.
        """
        if self.LOG_LEVELS.get(log.log_type, 0) < self.log_level:
            return

        with self.lock:
            self.logs.append(log.to_dict())

        # Print the log using its __str__ representation
        print(str(log))

    def add_log(self, log_type: str, data_type: str, cls_name: str, func_name: str, details: str):
        """
        Creates a Log object from individual components and adds it using add_log().
        """
        new_log = Log(log_type, data_type, cls_name, func_name, details)
        self.add_log_obj(new_log)

    def get_logs(self):
        """Returns all logs."""
        return self.logs

    def get_filtered_logs(self, start_time=None, end_time=None, log_type=None, data_type=None, cls_name=None, func_name=None):
        """
        Returns logs filtered by provided criteria.
        """
        with self.lock:
            filtered_logs = self.logs.copy()

            if start_time:
                start_time_dt = datetime.datetime.fromisoformat(start_time)
                filtered_logs = [
                    log for log in filtered_logs 
                    if datetime.datetime.fromisoformat(log["timestamp"]) >= start_time_dt
                ]

            if end_time:
                end_time_dt = datetime.datetime.fromisoformat(end_time)
                filtered_logs = [
                    log for log in filtered_logs 
                    if datetime.datetime.fromisoformat(log["timestamp"]) <= end_time_dt
                ]

            if log_type:
                filtered_logs = [log for log in filtered_logs if log["type"] == log_type.upper()]

            if data_type:
                filtered_logs = [log for log in filtered_logs if log["data_type"] == data_type]

            if cls_name:
                filtered_logs = [log for log in filtered_logs if log["class"] == cls_name]

            if func_name:
                filtered_logs = [log for log in filtered_logs if log["function"] == func_name]

        return filtered_logs
