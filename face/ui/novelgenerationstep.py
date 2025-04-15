from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea, QHBoxLayout
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
import torch as th
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from feat.plotting import plot_face
from io import BytesIO

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
        file_path = "data/situations.json"
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
            description = self.parent.manager_extraction.describe_face(aus)

            face_widget = QWidget()
            face_layout = QHBoxLayout()
            face_widget.setLayout(face_layout)

            pixmap = self.generate_face_pixmap(aus)

            img_label = QLabel()
            img_label.setPixmap(pixmap)
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            text_label = QLabel(f"<b>{i+1}. {situation}</b><br>{description}")
            text_label.setWordWrap(True)
            text_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

            face_layout.addWidget(img_label)
            face_layout.addWidget(text_label)
            self.output_layout.addWidget(face_widget)

    def generate_face_pixmap(self, au_values, size=(200, 200)):
        fig, ax = plt.subplots(figsize=(8, 9), dpi=400)
        plot_face(ax=ax, au=au_values)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=400, facecolor="white")
        buf.seek(0)

        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue(), "PNG")
        buf.close()
        plt.close(fig)

        return pixmap.scaled(size[0], size[1], Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
