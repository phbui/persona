from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea, QTextEdit
import json
import os
import numpy as np
import torch as th

class NovelGenerationStep(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.output_widget = QWidget()
        self.output_layout = QVBoxLayout()
        self.output_widget.setLayout(self.output_layout)
        self.scroll.setWidget(self.output_widget)

        self.layout.addWidget(QLabel("<h2>Novel Situations - Generated Faces</h2>"))
        self.layout.addWidget(self.scroll)

    def run_generation(self):
        file_path = "data/situations_novel.json"
        if not os.path.exists(file_path):
            self.output_layout.addWidget(QLabel("situations_novel.json not found."))
            return

        with open(file_path, "r", encoding="utf-8") as f:
            situations = json.load(f).get("situations", [])

        for i, situation in enumerate(situations):
            state = self.parent.manager_extraction.extract_features(situation)
            state += np.random.normal(0, 0.5, state.shape)
            state_tensor = th.tensor(state, dtype=th.float32).unsqueeze(0)
            state_tensor = state_tensor.to(next(self.parent.rl_model.policy.parameters()).device)

            action, _, _ = self.parent.rl_model.policy.select_action(state_tensor)
            aus = np.clip(action, 0, 3)
            desc = self.parent.manager_extraction.describe_face(aus)

            self.output_layout.addWidget(QLabel(f"<b>{i+1}. {situation}</b><br>{desc}<br><br>"))
