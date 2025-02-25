import json
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel, QLineEdit, QGroupBox, QFileDialog, QMessageBox
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
        self.save_policy_button = QPushButton("Save Policy")
        self.save_policy_button.clicked.connect(self.save_policy)
        policy_layout.addWidget(self.save_policy_button)
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

    def save_policy(self):
        print("Save Policy clicked.")
        QMessageBox.information(self, "Policy", "Policy saved (dummy function).")

    def download_policy(self):
        print("Download Policy clicked.")
        dummy_policy = {"policy": "dummy"}
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Policy as JSON", "", "JSON Files (*.json)")
        if file_path:
            with open(file_path, "w") as f:
                json.dump(dummy_policy, f)
            print(f"Policy downloaded to {file_path}.")
            QMessageBox.information(self, "Policy", f"Policy downloaded to {file_path} (dummy file).")

    def load_policy(self):
        print("Load Policy clicked.")
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Policy from JSON", "", "JSON Files (*.json)")
        if file_path:
            with open(file_path, "r") as f:
                policy_data = json.load(f)
            print("Loaded policy:", policy_data)
            QMessageBox.information(self, "Policy", f"Policy loaded from {file_path} (dummy load).")

    def create_mem_graph(self):
        print("Create Memory Graph clicked.")
        file_path, _ = QFileDialog.getOpenFileName(self, "Select .txt file for Memory Graph", "", "Text Files (*.txt)")
        if file_path:
            print(f"Selected .txt file: {file_path}")
            QMessageBox.information(self, "Memory Graph", f"Memory Graph created from {file_path} (dummy function).")

    def upload_mem_graph(self):
        print("Upload Memory Graph clicked.")
        file_path, _ = QFileDialog.getOpenFileName(self, "Select JSON file for Memory Graph", "", "JSON Files (*.json)")
        if file_path:
            print(f"Selected JSON file: {file_path}")
            QMessageBox.information(self, "Memory Graph", f"Memory Graph uploaded from {file_path} (dummy function).")

    def download_mem_graph(self):
        print("Download Memory Graph clicked.")
        dummy_mem_graph = {"memory_graph": "dummy"}
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Memory Graph as JSON", "", "JSON Files (*.json)")
        if file_path:
            with open(file_path, "w") as f:
                json.dump(dummy_mem_graph, f)
            print(f"Memory Graph downloaded to {file_path}.")
            QMessageBox.information(self, "Memory Graph", f"Memory Graph downloaded to {file_path} (dummy file).")

    def delete_mem_graph(self):
        print("Delete Memory Graph clicked.")
        QMessageBox.information(self, "Memory Graph", "Memory Graph deleted (dummy function).")

    def start_training(self):
        print("Start Training clicked.")
        epochs = self.epochs_entry.text()
        rounds = self.rounds_entry.text()
        clip_range = self.clip_range_entry.text()
        learning_rate = self.learning_rate_entry.text()
        discount_factor = self.discount_factor_entry.text()
        gae_param = self.gae_entry.text()
        print("Starting training with parameters:")
        print(f"Epochs: {epochs}")
        print(f"Rounds: {rounds}")
        print(f"Clip Range: {clip_range}")
        print(f"Learning Rate: {learning_rate}")
        print(f"Discount Factor: {discount_factor}")
        print(f"GAE Parameter: {gae_param}")
        QMessageBox.information(self, "Training", "Training started (dummy function).")
