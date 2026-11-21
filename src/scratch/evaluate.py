
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from torch.utils.data import DataLoader

from .data import (
    ScratchDataset,
    load_samples,
    MAX_LENGTH,
    IMAGE_SIZE,
    seed_everything,
)
from .models import MODELS


RESULTS_DIR = Path("results/scratch")
CHECKPOINT_DIR = Path("checkpoints/scratch")
LABEL_NAMES = ["negative", "neutral", "positive"]


def load_checkpoint(path, device):
    return torch.load(
        path,
        map_location=device,
        weights_only=True,
    )


def evaluate_one(name, loader, vocab_size, device):
    path = CHECKPOINT_DIR / f"{name}_full.pt"

    if not path.is_file():
        raise FileNotFoundError(path)

    checkpoint = load_checkpoint(path, device)

    if checkpoint["model_name"] != name:
        raise ValueError(f"Model mismatch: {path}")

    if checkpoint["vocab_size"] != vocab_size:
        raise ValueError(f"Vocabulary mismatch: {path}")

    if checkpoint["limit_per_split"] != 0:
        raise ValueError(f"Not a full-data checkpoint: {path}")

    if checkpoint["max_length"] != MAX_LENGTH:
        raise ValueError("Text length mismatch")

    if checkpoint["image_size"] != IMAGE_SIZE:
        raise ValueError("Image size mismatch")

    model = MODELS[name](vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true = []
    y_pred = []

    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("label")

            inputs = {
                key: value.to(device)
                for key, value in batch.items()
            }

            logits = model(inputs)
            predictions = logits.argmax(dim=1)

            y_true.extend(labels.tolist())
            y_pred.extend(predictions.cpu().tolist())

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2],
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=LABEL_NAMES,
        zero_division=0,
        output_dict=True,
    )

    metrics = {
        "model": name,
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_val_macro_f1": float(
            checkpoint["best_val_macro_f1"]
        ),
        "test_samples": len(y_true),
    }

    return metrics, cm.tolist(), report


def main():
    seed_everything(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n===== SCRATCH TEST EVALUATION =====")
    print("Device:", device)

    with (RESULTS_DIR / "vocab.json").open(
        encoding="utf-8"
    ) as file:
        vocab = json.load(file)

    df = load_samples()
    test_df = df[df["split"] == "test"]

    print("Test samples:", len(test_df))
    print("Vocabulary size:", len(vocab))

    dataset = ScratchDataset(
        test_df,
        vocab,
        max_length=MAX_LENGTH,
        image_size=IMAGE_SIZE,
    )

    # Conservative batch size for the RTX 2050.
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    all_metrics = []
    all_details = {}

    for name in MODELS:
        print(f"\nEvaluating: {name}", flush=True)

        metrics, cm, report = evaluate_one(
            name,
            loader,
            len(vocab),
            device,
        )

        all_metrics.append(metrics)

        all_details[name] = {
            "metrics": metrics,
            "confusion_matrix": cm,
            "classification_report": report,
        }

        print(
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Macro-F1: {metrics['macro_f1']:.4f}"
        )

    output_csv = RESULTS_DIR / "full_test_metrics.csv"
    output_json = RESULTS_DIR / "full_test_details.json"

    # Avoid silently replacing a previous evaluation.
    for path in (output_csv, output_json):
        if path.exists():
            raise FileExistsError(
                f"Output already exists: {path}"
            )

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(all_metrics[0].keys()),
        )
        writer.writeheader()
        writer.writerows(all_metrics)

    output_json.write_text(
        json.dumps(all_details, indent=2),
        encoding="utf-8",
    )

    print("\n===== FINAL TEST RESULTS =====")

    for metrics in all_metrics:
        print(
            f"{metrics['model']:8s} | "
            f"Accuracy={metrics['accuracy']:.4f} | "
            f"Precision={metrics['macro_precision']:.4f} | "
            f"Recall={metrics['macro_recall']:.4f} | "
            f"Macro-F1={metrics['macro_f1']:.4f}"
        )

    print("\nSaved:", output_csv)
    print("Saved:", output_json)
    print("\nEVALUATION COMPLETE")


if __name__ == "__main__":
    main()
