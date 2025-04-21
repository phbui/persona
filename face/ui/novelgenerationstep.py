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

            # Generate emotion scores from AUs
            emotion_scores = self._emotion_scores_from_aus(aus)
            # Format emotion percentages
            emotion_desc = ", ".join(f"{emo}: {score:.2f}" for emo, score in emotion_scores.items())
            # Generate detailed AU description
            au_description = self.parent.manager_extraction.describe_face(aus)
            # Combine for final description
            description = f"{emotion_desc} — {au_description}"

            face_widget = QWidget()
            face_layout = QHBoxLayout()
            face_widget.setLayout(face_layout)

            pixmap = self.generate_face_pixmap(aus)

            img_label = QLabel()
            img_label.setPixmap(pixmap)
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            text_label = QLabel(f"<b>{i+1}. {situation}")
            #text_label = QLabel(f"<b>{i+1}. {situation}</b><br>{description}")
            text_label.setWordWrap(True)
            text_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

            face_layout.addWidget(img_label)
            face_layout.addWidget(text_label)
            self.output_layout.addWidget(face_widget)

    def _emotion_scores_from_aus(self, aus):
        """
        Rule-based scoring for basic emotions based on AU intensities.
        Returns a dict mapping emotion to a 0-1 score.
        """
        # Normalize intensities (0-3) to 0-1
        norm = lambda x: min(max(x / 3.0, 0.0), 1.0)

        scores = {}
        # Happiness: AU6 (index 5) & AU12 (index 11)
        scores['Happiness'] = min(norm(aus[5]), norm(aus[11]))
        # Sadness: AU1 (0), AU4 (3) & AU15 (14)
        sadness_vals = [norm(aus[0]), norm(aus[3])]
        if len(aus) > 14:
            sadness_vals.append(norm(aus[14]))
        scores['Sadness'] = min(sadness_vals)
        # Anger: AU4 (3) & AU7 (6)
        anger_vals = [norm(aus[3])]
        if len(aus) > 6:
            anger_vals.append(norm(aus[6]))
        scores['Anger'] = min(anger_vals)
        # Surprise: AU1 (0), AU2 (1) & AU5 (4)
        scores['Surprise'] = min(norm(aus[0]), norm(aus[1]), norm(aus[4]))
        # Neutral as inverse of strongest emotion
        strongest = max(scores.values())
        scores['Neutral'] = max(0.0, 1.0 - strongest)

        return scores

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
