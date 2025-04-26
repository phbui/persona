import datetime
import json

class Log:
    """
    Represents a single log entry.
    """
    def __init__(self, log_type, data_type, cls_name, func_name, details, timestamp=None):
        self.timestamp = timestamp if timestamp else datetime.datetime.now()
        self.log_type = log_type.upper()
        self.data_type = data_type
        self.cls_name = cls_name
        self.func_name = func_name
        self.details = details

    def to_dict(self):
        """
        Converts the log entry to a dictionary.
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "type": self.log_type,
            "data_type": self.data_type,
            "class": self.cls_name,
            "function": self.func_name,
            "details": self.details,
        }

    def __str__(self):
        """
        Returns the string representation of the log entry.
        """
        return json.dumps(self.to_dict(), indent=4)
