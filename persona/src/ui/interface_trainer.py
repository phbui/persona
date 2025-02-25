import json
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel, QLineEdit, QGroupBox, QFileDialog, QMessageBox, QDoubleValidator
from training.trainer import Trainer

class Interface_Trainer(QWidget):
    def __init__(self):
        super().__init__()
        self.trainer = Trainer()
        self._setup_ui()

    def _setup_ui(self):
        self.main_layout = QVBoxLayout(self)
        hyper_group = QGroupBox("Hyperparameters")
        hyper_layout = QGridLayout()
        self.label_epochs = QLabel("Number of Epochs:")
        self.epochs_entry = QLineEdit()
        hyper_layout.addWidget(self.label_epochs, 0, 0)
        hyper_layout.addWidget(self.epochs_entry, 0, 1)
        self.label_rounds = QLabel("Number of Rounds:")
        self.rounds_entry = QLineEdit()
        hyper_layout.addWidget(self.label_rounds, 1, 0)
        hyper_layout.addWidget(self.rounds_entry, 1, 1)
        self.label_clip = QLabel("Clip Range (ε):")
        self.clip_range_entry = QLineEdit()
        hyper_layout.addWidget(self.label_clip, 2, 0)
        hyper_layout.addWidget(self.clip_range_entry, 2, 1)
        self.label_lr = QLabel("Learning Rate:")
        self.learning_rate_entry = QLineEdit()
        hyper_layout.addWidget(self.label_lr, 3, 0)
        hyper_layout.addWidget(self.learning_rate_entry, 3, 1)
        self.label_discount = QLabel("Discount Factor (γ):")
        self.discount_factor_entry = QLineEdit()
        hyper_layout.addWidget(self.label_discount, 4, 0)
        hyper_layout.addWidget(self.discount_factor_entry, 4, 1)
        self.label_gae = QLabel("GAE Parameter (λ):")
        self.gae_entry = QLineEdit()
        hyper_layout.addWidget(self.label_gae, 5, 0)
        hyper_layout.addWidget(self.gae_entry, 5, 1)
        hyper_group.setLayout(hyper_layout)
        self.main_layout.addWidget(hyper_group)
        policy_group = QGroupBox("Policy")
        policy_layout = QHBoxLayout()
        self.download_policy_button = QPushButton("Download Policy (to JSON)")
        self.download_policy_button.clicked.connect(self.download_policy)
        policy_layout.addWidget(self.download_policy_button)
        self.load_policy_button = QPushButton("Load Policy (from JSON)")
        self.load_policy_button.clicked.connect(self.load_policy)
        policy_layout.addWidget(self.load_policy_button)
        policy_group.setLayout(policy_layout)
        self.main_layout.addWidget(policy_group)
        mem_group = QGroupBox("Memory Graph")
        mem_layout = QHBoxLayout()
        self.create_mem_graph_button = QPushButton("Create Memory Graph (from .txt)")
        self.create_mem_graph_button.clicked.connect(self.create_mem_graph)
        mem_layout.addWidget(self.create_mem_graph_button)
        self.upload_mem_graph_button = QPushButton("Upload Memory Graph (from JSON)")
        self.upload_mem_graph_button.clicked.connect(self.upload_mem_graph)
        mem_layout.addWidget(self.upload_mem_graph_button)
        self.download_mem_graph_button = QPushButton("Download Memory Graph (to JSON)")
        self.download_mem_graph_button.clicked.connect(self.download_mem_graph)
        mem_layout.addWidget(self.download_mem_graph_button)
        self.delete_mem_graph_button = QPushButton("Delete Memory Graph")
        self.delete_mem_graph_button.clicked.connect(self.delete_mem_graph)
        mem_layout.addWidget(self.delete_mem_graph_button)
        mem_group.setLayout(mem_layout)
        self.main_layout.addWidget(mem_group)
        self.start_training_button = QPushButton("Start Training")
        self.start_training_button.clicked.connect(self.start_training)
        self.main_layout.addWidget(self.start_training_button)
        self.setLayout(self.main_layout)

        self.epochs_entry.setValidator(QDoubleValidator())
        self.rounds_entry.setValidator(QDoubleValidator())
        self.clip_range_entry.setValidator(QDoubleValidator())
        self.learning_rate_entry.setValidator(QDoubleValidator())
        self.discount_factor_entry.setValidator(QDoubleValidator())
        self.gae_entry.setValidator(QDoubleValidator())

    def download_policy(self):
        file_path, file_name = QFileDialog.getSaveFileName(self, "Save Policy as JSON", "", "JSON Files (*.json)")
        if file_path:
            self.trainer.download_policy(file_path, file_name)
            QMessageBox.information(self, "Policy", f"Policy downloaded to {file_path} (dummy file).")

    def load_policy(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Policy from JSON", "", "JSON Files (*.json)")
        if file_path:
            self.trainer.load_policy(file_path)
            QMessageBox.information(self, "Policy", f"Policy loaded from {file_path} (dummy load).")

    def create_mem_graph(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select .txt file for Memory Graph", "", "Text Files (*.txt)")
        if file_path:
            self.trainer.create_graph(file_path)
            QMessageBox.information(self, "Memory Graph", f"Memory Graph created from {file_path} (dummy function).")

    def upload_mem_graph(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select JSON file for Memory Graph", "", "JSON Files (*.json)")
        if file_path:
            self.trainer.upload_graph(file_path)
            QMessageBox.information(self, "Memory Graph", f"Memory Graph uploaded from {file_path} (dummy function).")

    def download_mem_graph(self):
        file_path, file_name = QFileDialog.getSaveFileName(self, "Save Memory Graph as JSON", "", "JSON Files (*.json)")
        if file_path:
            self.trainer.download_graph(file_path, file_name)
            QMessageBox.information(self, "Memory Graph", f"Memory Graph downloaded to {file_path} (dummy file).")

    def delete_mem_graph(self):
        QMessageBox.information(self, "Memory Graph", "Memory Graph deleted.")
        self.trainer.delete_graph()

    def start_training(self):
        try:
            epochs = int(self.epochs_entry.text())
            rounds = int(self.rounds_entry.text())
            clip_range = float(self.clip_range_entry.text())
            learning_rate = float(self.learning_rate_entry.text())
            discount_factor = float(self.discount_factor_entry.text())
            gae_param = float(self.gae_entry.text())

            if not (0 < clip_range <= 1):
                raise ValueError("Clip range must be between 0 and 1.")
            if not (0 < learning_rate <= 1):
                raise ValueError("Learning rate must be between 0 and 1.")
            if not (0 < discount_factor <= 1):
                raise ValueError("Discount factor must be between 0 and 1.")
            if not (0 < gae_param <= 1):
                raise ValueError("GAE parameter must be between 0 and 1.")

            self.trainer.train(epochs, rounds, clip_range, learning_rate, discount_factor, gae_param)
            QMessageBox.information(self, "Training", "Training started successfully.")

        except ValueError as e:
            QMessageBox.warning(self, "Input Error", f"Invalid input: {e}")
