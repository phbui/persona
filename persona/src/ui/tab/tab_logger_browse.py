from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QPushButton, QLineEdit
from ..interface import Interface
from ...log.logger import Logger

class Tab_Logger_Browse(Interface):
    """
    A separate tab for browsing logs.
    This class can be directly added to a QTabWidget (e.g., via addTab(Tab_Logger_Browse(), "Logger")).
    It displays logs in a table, offers search/filter capabilities, and allows downloading of the displayed logs.
    """
    def __init__(self):
        # Initialize the base Interface with a title and description.
        super().__init__(title="Browse", description="Browse and filter logs.")
        self.logger = Logger()
        self.current_logs = self.logger.get_logs()
        self.init_ui_components()

    def init_ui_components(self):
        """
        Sets up the UI components specific to the Logger browsing tab.
        """
        # Create a new layout that builds on the base interface layout.
        main_layout = self.layout()

        # Create a horizontal layout for search, filter, and download controls.
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search logs...")
        search_button = QPushButton("Filter")
        search_button.clicked.connect(self.filter_logs)
        download_button = QPushButton("Download Displayed Logs")
        download_button.clicked.connect(self.download_displayed_logs)
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(search_button)
        search_layout.addWidget(download_button)

        # Create a table widget for displaying log entries.
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["Timestamp", "Type", "Data Type", "Class", "Function", "Details"])
        self.table.horizontalHeader().setStretchLastSection(True)

        # Add the search layout and table widget to the main layout.
        main_layout.addLayout(search_layout)
        main_layout.addWidget(self.table)

        # Populate the table with the initial set of logs.
        self.populate_table(self.current_logs)

    def populate_table(self, logs):
        """
        Populates the log table with the provided list of logs.
        """
        self.table.setRowCount(0)
        for log in logs:
            row_position = self.table.rowCount()
            self.table.insertRow(row_position)
            self.table.setItem(row_position, 0, QTableWidgetItem(log.get("timestamp", "")))
            self.table.setItem(row_position, 1, QTableWidgetItem(log.get("type", "")))
            self.table.setItem(row_position, 2, QTableWidgetItem(log.get("data_type", "")))
            self.table.setItem(row_position, 3, QTableWidgetItem(log.get("class", "")))
            self.table.setItem(row_position, 4, QTableWidgetItem(log.get("function", "")))
            self.table.setItem(row_position, 5, QTableWidgetItem(log.get("details", "")))

    def filter_logs(self):
        """
        Filters logs based on the search input across all fields.
        """
        search_text = self.search_input.text().lower()
        all_logs = self.logger.get_logs()
        if not search_text:
            filtered_logs = all_logs
        else:
            filtered_logs = [
                log for log in all_logs
                if (search_text in log.get("timestamp", "").lower() or
                    search_text in log.get("type", "").lower() or
                    search_text in log.get("data_type", "").lower() or
                    search_text in log.get("class", "").lower() or
                    search_text in log.get("function", "").lower() or
                    search_text in log.get("details", "").lower())
            ]
        self.current_logs = filtered_logs
        self.populate_table(filtered_logs)

    def download_displayed_logs(self):
        """
        Downloads the logs currently displayed in the table using the common download_file method from the base interface.
        """
        # The base Interface.download_file opens a file dialog to let the user choose file name/location.
        self.download_file(self.current_logs)
