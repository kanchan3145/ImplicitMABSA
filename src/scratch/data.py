
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


SEED = 42
MAX_LENGTH = 64
IMAGE_SIZE = 128
MAX_VOCAB = 20000
MIN_FREQ = 2

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

DATA_PATH = Path("results/samples.csv")
OUTPUT_DIR = Path("results/scratch")
VOCAB_PATH = OUTPUT_DIR / "vocab.json"


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize(text):
    """
    Simple tokenizer with explicit preservation
    of the masked-aspect placeholder.
    """
    text = str(text).lower()
    text = text.replace(
        "[masked_aspect]",
        " maskedaspecttoken "
    )

    return re.findall(
        r"[a-z0-9]+(?:'[a-z0-9]+)?|[^\w\s]",
        text
    )


def load_samples(path=DATA_PATH):
    df = pd.read_csv(path)

    required = {
        "text", "image", "label", "split"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing dataset columns: {missing}"
        )

    if df[list(required)].isna().any().any():
        raise ValueError("Dataset contains missing values")

    if not df["label"].isin([0, 1, 2]).all():
        raise ValueError("Invalid sentiment labels")

    if not df["split"].isin(
        ["train", "val", "test"]
    ).all():
        raise ValueError("Unexpected dataset split")

    for split in ("train", "val", "test"):
        if not (df["split"] == split).any():
            raise ValueError(f"Empty {split} split")

    return df


def build_vocabulary(
    train_texts,
    max_vocab=MAX_VOCAB,
    min_freq=MIN_FREQ
):
    counter = Counter()

    for text in train_texts:
        counter.update(tokenize(text))

    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1
    }

    words = sorted(
        counter.items(),
        key=lambda item: (-item[1], item[0])
    )

    for word, freq in words:
        if freq < min_freq:
            continue

        if len(vocab) >= max_vocab:
            break

        vocab[word] = len(vocab)

    return vocab


def encode_text(
    text,
    vocab,
    max_length=MAX_LENGTH
):
    tokens = tokenize(text)[:max_length]

    ids = [
        vocab.get(token, vocab[UNK_TOKEN])
        for token in tokens
    ]

    mask = [1] * len(ids)

    padding = max_length - len(ids)

    ids += [vocab[PAD_TOKEN]] * padding
    mask += [0] * padding

    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(mask, dtype=torch.bool)
    )


class ScratchDataset(Dataset):
    def __init__(
        self,
        df,
        vocab,
        max_length=MAX_LENGTH,
        image_size=IMAGE_SIZE
    ):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.max_length = max_length
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        input_ids, attention_mask = encode_text(
            row["text"],
            self.vocab,
            self.max_length
        )

        with Image.open(row["image"]) as img:
            img = img.convert("RGB")
            img = img.resize(
                (self.image_size, self.image_size),
                Image.Resampling.BILINEAR
            )
            image_array = np.asarray(
                img, dtype=np.float32
            ).copy()

        image = torch.from_numpy(
            image_array
        ).permute(2, 0, 1) / 255.0

        # Normalize RGB values to [-1, 1].
        image = (image - 0.5) / 0.5

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image,
            "label": torch.tensor(
                int(row["label"]),
                dtype=torch.long
            )
        }


def create_loaders(
    df,
    vocab,
    batch_size=16,
    max_length=MAX_LENGTH,
    image_size=IMAGE_SIZE
):
    loaders = {}

    for split in ("train", "val", "test"):
        part = df[df["split"] == split]

        dataset = ScratchDataset(
            part,
            vocab,
            max_length,
            image_size
        )

        generator = torch.Generator()
        generator.manual_seed(SEED)

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            generator=generator,
            pin_memory=torch.cuda.is_available()
        )

    return loaders


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch",
        type=int,
        default=16
    )

    args = parser.parse_args()

    if args.batch < 1:
        raise ValueError("Batch size must be positive")

    seed_everything()

    df = load_samples()

    train_df = df[df["split"] == "train"]

    vocab = build_vocabulary(
        train_df["text"]
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if VOCAB_PATH.exists():
        existing = json.loads(
            VOCAB_PATH.read_text(encoding="utf-8")
        )
        if existing != vocab:
            raise RuntimeError(
                "Existing vocabulary differs from the "
                "current training dataset. Preserve the "
                "old experiment before rebuilding."
            )
    else:
        VOCAB_PATH.write_text(
            json.dumps(
                vocab,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

    loaders = create_loaders(
        df,
        vocab,
        batch_size=args.batch
    )

    print("\n===== SCRATCH DATA PIPELINE =====")

    print("Vocabulary size:", len(vocab))
    print("Maximum text length:", MAX_LENGTH)
    print("Image size:", IMAGE_SIZE)
    print("Vocabulary source: TRAIN ONLY")

    print("\nDataset sizes:")

    for split, loader in loaders.items():
        print(split, len(loader.dataset))

    print("\nChecking one training batch...")

    batch = next(iter(loaders["train"]))

    for key, value in batch.items():
        print(
            key,
            tuple(value.shape),
            value.dtype
        )

    assert batch["input_ids"].shape[1] == MAX_LENGTH
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["image"].shape[1:] == (
        3, IMAGE_SIZE, IMAGE_SIZE
    )

    assert batch["label"].min().item() >= 0
    assert batch["label"].max().item() <= 2

    print("\nMasked placeholder token ID:")
    print(vocab.get("maskedaspecttoken"))

    assert "maskedaspecttoken" in vocab

    print("\nPIPELINE TEST PASSED")


if __name__ == "__main__":
    main()
