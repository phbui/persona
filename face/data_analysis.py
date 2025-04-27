#!/usr/bin/env python3
import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def select_file(directory="data/responses"):
    csvs = sorted(glob.glob(os.path.join(directory, "*.csv")))
    if not csvs:
        raise FileNotFoundError(f"No CSVs in {directory}")
    print("Available CSVs:")
    for i, path in enumerate(csvs):
        print(f"  [{i}] {os.path.basename(path)}")
    idx = int(input("Select file number: "))
    return csvs[idx]

def load_face_data(csv_path):
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f]

    face_rows = []
    for i, line in enumerate(lines):
        if line.startswith("Rate the faces"):
            j = i + 1
            while j < len(lines) and lines[j].startswith("Face"):
                face_rows.append(lines[j])
                j += 1

    if not face_rows:
        raise ValueError("No 'Face' blocks found in file")

    df = pd.DataFrame([r.split(',') for r in face_rows],
                      columns=["Face", "Avg", "Min", "Max", "Count"])

    # Rename faces
    df["Face"] = df["Face"].replace({
        "Face 1": "HF",
        "Face 2": "HF + LLM",
        "Face 3": "LLM"
    })

    for col in ["Avg", "Min", "Max", "Count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(1)
    return df

def analyze_and_plot(faces_df):
    stats = faces_df.groupby("Face").agg(
        avg_of_avgs=("Avg", "mean"),
        overall_min=("Min", "min"),
        overall_max=("Max", "max"),
        total_counts=("Count", "sum"),
        std_of_avgs=("Avg", "std")
    )
    print("\n=== Aggregate Stats by Face ===")
    print(stats.to_string())

    weighted_mean = faces_df.groupby("Face").apply(
        lambda g: np.average(g["Avg"], weights=g["Count"])
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    faces_df.boxplot(
        column="Avg",
        by="Face",
        grid=False,
        ax=ax1,
        showfliers=False
    )
    ax1.set_title("Per-Question Avg Ratings by Face")
    ax1.set_xlabel("Face Generation Version")
    ax1.set_ylabel("Avg Rating (1–5)")
    fig.suptitle("")

    weighted_mean.plot(
        kind="bar",
        ax=ax2,
        legend=False
    )
    ax2.set_title("Weighted Overall Mean Rating")
    ax2.set_xlabel("Face Generation Version")
    ax2.set_ylabel("Weighted Mean")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    path = select_file()
    df_faces = load_face_data(path)
    analyze_and_plot(df_faces)
