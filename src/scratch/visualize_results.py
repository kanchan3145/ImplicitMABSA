
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V2_PATH = Path("results/comparison.csv")
SCRATCH_PATH = Path("results/scratch/full_test_metrics.csv")
DETAILS_PATH = Path("results/scratch/full_test_details.json")
OUTPUT_DIR = Path("results/figures")

MODEL_ORDER = ["text", "image", "concat", "cross"]
MODEL_LABELS = [
    "Text-only",
    "Image-only",
    "Concatenation",
    "Cross-attention",
]
CLASS_NAMES = ["Negative", "Neutral", "Positive"]


def load_results():
    v2 = pd.read_csv(V2_PATH).set_index("model")
    scratch = pd.read_csv(SCRATCH_PATH).set_index("model")

    scratch = scratch.rename(index={"bilstm": "text", "cnn": "image"})

    for name, frame in [("V2", v2), ("Scratch", scratch)]:
        missing = set(MODEL_ORDER) - set(frame.index)
        if missing:
            raise ValueError(f"{name} missing models: {missing}")

    with DETAILS_PATH.open(encoding="utf-8") as file:
        details = json.load(file)

    return (
        v2.loc[MODEL_ORDER],
        scratch.loc[MODEL_ORDER],
        details,
    )


def save_figure(filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", path)


def comparison_chart(v2, scratch, metric, ylabel, filename):
    x = np.arange(len(MODEL_ORDER))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6))

    bars1 = ax.bar(
        x - width / 2,
        v2[metric].values,
        width,
        label="Pretrained V2",
    )

    bars2 = ax.bar(
        x + width / 2,
        scratch[metric].values,
        width,
        label="Scratch",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 0.8)
    ax.set_title(f"Pretrained vs Scratch — {ylabel}")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    for bars in [bars1, bars2]:
        ax.bar_label(bars, fmt="%.3f", padding=3)

    fig.tight_layout()
    save_figure(filename)


def classwise_f1_chart(details):
    names = ["bilstm", "cnn", "concat", "cross"]
    labels = ["BiLSTM", "CNN", "Concat", "Cross"]

    values = np.array([
        [
            details[name]["classification_report"][label]["f1-score"]
            for label in ["negative", "neutral", "positive"]
        ]
        for name in names
    ])

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, class_name in enumerate(CLASS_NAMES):
        bars = ax.bar(
            x + (i - 1) * width,
            values[:, i],
            width,
            label=class_name,
        )
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("F1-score")
    ax.set_title("Scratch Models — Class-wise F1")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    save_figure("scratch_classwise_f1.png")


def confusion_matrix_chart(details, model_name):
    matrix = np.array(
        details[model_name]["confusion_matrix"],
        dtype=int,
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)

    ax.set_xlabel("Predicted sentiment")
    ax.set_ylabel("Actual sentiment")
    ax.set_title(f"{model_name.upper()} — Confusion Matrix")

    threshold = matrix.max() / 2

    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )

    fig.tight_layout()
    save_figure(f"scratch_{model_name}_confusion_matrix.png")


def main():
    v2, scratch, details = load_results()

    print("===== GENERATING RESULT FIGURES =====")

    comparison_chart(
        v2,
        scratch,
        "accuracy",
        "Test Accuracy",
        "pretrained_vs_scratch_accuracy.png",
    )

    comparison_chart(
        v2,
        scratch,
        "macro_f1",
        "Test Macro-F1",
        "pretrained_vs_scratch_macro_f1.png",
    )

    classwise_f1_chart(details)

    for name in ["bilstm", "cnn", "concat", "cross"]:
        confusion_matrix_chart(details, name)

    print("\nVISUALIZATION COMPLETE")


if __name__ == "__main__":
    main()
