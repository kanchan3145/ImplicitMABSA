
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from .data import (
    SEED,
    MAX_LENGTH,
    IMAGE_SIZE,
    ScratchDataset,
    load_samples,
    seed_everything,
)
from .models import MODELS


VOCAB_PATH = Path("results/scratch/vocab.json")
CHECKPOINT_DIR = Path("checkpoints/scratch")
RESULTS_DIR = Path("results/scratch")


def prepare_data(limit, batch_size):
    df = load_samples()

    if limit > 0:
        # Sample each split independently and deterministically.
        # Preserve the original split assignments.
        parts = []
        for split in ("train", "val", "test"):
            part = df[df["split"] == split]
            n = min(limit, len(part))
            parts.append(
                part.sample(n=n, random_state=SEED)
            )

        import pandas as pd
        df = pd.concat(parts, ignore_index=True)

    with VOCAB_PATH.open(encoding="utf-8") as file:
        vocab = json.load(file)

    loaders = {}

    for split in ("train", "val", "test"):
        part = df[df["split"] == split]

        dataset = ScratchDataset(
            part,
            vocab,
            max_length=MAX_LENGTH,
            image_size=IMAGE_SIZE
        )

        generator = torch.Generator()
        generator.manual_seed(SEED)

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            generator=generator
        )

    return df, vocab, loaders


def move_batch(batch, device):
    return {
        key: value.to(
            device,
            non_blocking=(device.type == "cuda")
        )
        for key, value in batch.items()
    }


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0
            )
        )
    }


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_true = []
    all_pred = []

    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)

            logits = model(batch)
            labels = batch["label"]

            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size

            all_true.extend(
                labels.cpu().tolist()
            )
            all_pred.extend(
                logits.argmax(dim=1).cpu().tolist()
            )

    metrics = calculate_metrics(all_true, all_pred)
    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):
    model.train()

    total_loss = 0.0

    for batch in loader:
        batch = move_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(batch)
        labels = batch["label"]

        loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")

        loss.backward()

        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += (
            loss.item() * labels.size(0)
        )

    return total_loss / len(loader.dataset)


def run(args):
    seed_everything(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n===== SCRATCH TRAINING =====")
    print("Model:", args.model)
    print("Device:", device)
    print("Seed:", SEED)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    df, vocab, loaders = prepare_data(
        args.limit,
        args.batch
    )

    print("Vocabulary size:", len(vocab))

    for split in ("train", "val", "test"):
        print(
            f"{split}:",
            len(loaders[split].dataset)
        )

    model = MODELS[args.model](
        vocab_size=len(vocab)
    ).to(device)

    train_labels = df.loc[
        df["split"] == "train", "label"
    ].to_numpy()

    counts = np.bincount(
        train_labels,
        minlength=3
    )

    # In a small smoke-test sample, a class may be absent.
    # Give absent classes zero weight instead of dividing by zero.
    weights = np.zeros(3, dtype=np.float32)
    present = counts > 0

    weights[present] = (
        len(train_labels)
        / (present.sum() * counts[present])
    )

    print("Training label counts:", counts.tolist())
    print("Class weights:", weights.tolist())

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            weights,
            dtype=torch.float32,
            device=device
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    run_name = (
        f"{args.model}_smoke"
        if args.limit > 0
        else f"{args.model}_full"
    )

    checkpoint_path = (
        CHECKPOINT_DIR / f"{run_name}.pt"
    )
    history_path = (
        RESULTS_DIR / f"{run_name}_history.json"
    )

    # Never overwrite an existing experiment.
    if checkpoint_path.exists() or history_path.exists():
        raise FileExistsError(
            f"Existing experiment output for {run_name}. "
            "Preserve or rename it before rerunning."
        )

    best_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device
        )

        val_metrics = evaluate(
            model,
            loaders["val"],
            criterion,
            device
        )

        record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"]
        }

        history.append(record)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}",
            flush=True
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model,
                    "vocab_size": len(vocab),
                    "seed": SEED,
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_f1,
                    "limit_per_split": args.limit,
                    "max_length": MAX_LENGTH,
                    "image_size": IMAGE_SIZE
                },
                checkpoint_path
            )

            print(
                "  Saved best checkpoint:",
                checkpoint_path
            )
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print("Early stopping triggered")
            break

    history_path.write_text(
        json.dumps(history, indent=2),
        encoding="utf-8"
    )

    print("\n===== TRAINING COMPLETE =====")
    print("Best epoch:", best_epoch)
    print("Best validation macro-F1:", round(best_f1, 4))
    print("Checkpoint:", checkpoint_path)
    print("History:", history_path)

    # Do not evaluate on the test split here.
    # The held-out test set is reserved for final evaluation.


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=list(MODELS),
        default="bilstm"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=2
    )

    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be >= 0")
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.batch < 1:
        parser.error("--batch must be >= 1")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.patience < 1:
        parser.error("--patience must be >= 1")

    run(args)


if __name__ == "__main__":
    main()
