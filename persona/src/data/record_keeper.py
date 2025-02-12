class RecordKeeper:
    _instance = None

    def __init__(self):
        self.records = []

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, record):
        self.records.append(record)

    def get_all_records(self):
        return self.records