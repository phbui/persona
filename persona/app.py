import sys
import signal
from PyQt6.QtWidgets import QApplication
from src.ui.manager_tab import Manager_Tab

def main():
    # Let Ctrl+C (SIGINT) terminate the program instead of ignoring it (on some platforms).
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)

    # Access the singleton UI
    ui = Manager_Tab()
    ui.show()

    # (Optional) Start on a particular tab, e.g., Logger
    ui.switch_tab(1)

    # This blocks until the main window is closed or sys.exit is called
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
