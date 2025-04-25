import os
import re
import json
import matplotlib.pyplot as plt

def load_best_metrics_for_model(model_path):
    ckpts = [
        d for d in os.listdir(model_path)
        if os.path.isdir(os.path.join(model_path, d))
        and re.fullmatch(r"checkpoint-(\d+)", d)
    ]
    ckpts.sort(key=lambda d: int(d.split("-", 1)[1]))

    nums = []
    metrics = []
    for ckpt in ckpts:
        state_file = os.path.join(model_path, ckpt, "trainer_state.json")
        if not os.path.isfile(state_file):
            continue
        with open(state_file, "r") as f:
            state = json.load(f)
        best = state.get("best_metric")
        if best is None:
            continue
        nums.append(int(ckpt.split("-", 1)[1]))
        metrics.append(best)

    return nums, metrics

if __name__ == "__main__":
    models_root = "models/llm"
    model_names = [
        d for d in os.listdir(models_root)
        if os.path.isdir(os.path.join(models_root, d))
    ]
    if not model_names:
        print("No models found in", models_root)
        exit(1)

    print("Available models:")
    for idx, name in enumerate(model_names):
        print(f"[{idx}] {name}")
    sel = input("Select model by number: ")
    try:
        sel = int(sel)
        model_name = model_names[sel]
    except (ValueError, IndexError):
        print("Invalid selection.")
        exit(1)

    model_path = os.path.join(models_root, model_name)
    print(f"\nLoading checkpoints for model: {model_name}\n")

    checkpoints, best_metrics = load_best_metrics_for_model(model_path)
    if not checkpoints:
        print("No checkpoints or no best_metric found.")
        exit(1)

    print(f"Checkpoints: {checkpoints}")
    print(f"Best Metrics: {best_metrics}")

    plt.figure(figsize=(8, 5))
    plt.plot(checkpoints, best_metrics, marker="o", linestyle="-")
    plt.xlabel("Checkpoint Number")
    plt.ylabel("Best Metric")
    plt.title(f"Best Metric over Checkpoints for Fine-Tuned LLM'")
    plt.grid(True)
    plt.show()
