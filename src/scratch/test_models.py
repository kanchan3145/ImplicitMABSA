
import json

import torch
from torch import nn

from .models import MODELS


def main():
    torch.manual_seed(42)

    with open(
        "results/scratch/vocab.json",
        encoding="utf-8"
    ) as file:
        vocab = json.load(file)

    vocab_size = len(vocab)

    # Use a small synthetic batch for CPU verification.
    batch_size = 2
    sequence_length = 64

    batch = {
        "input_ids": torch.randint(
            2, vocab_size,
            (batch_size, sequence_length)
        ),
        "attention_mask": torch.ones(
            batch_size,
            sequence_length,
            dtype=torch.bool
        ),
        "image": torch.randn(
            batch_size, 3, 128, 128
        )
    }

    # Simulate text padding.
    batch["attention_mask"][0, 20:] = False
    batch["input_ids"][0, 20:] = 0

    print("\n===== SCRATCH MODEL VERIFICATION =====")
    print("Vocabulary size:", vocab_size)

    for name, model_class in MODELS.items():
        print(f"\nTesting: {name}")

        model = model_class(vocab_size)
        model.train()

        logits = model(batch)

        print("Output shape:", tuple(logits.shape))

        assert logits.shape == (batch_size, 3)
        assert torch.isfinite(logits).all()

        labels = torch.tensor([0, 2])

        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()

        params = sum(
            p.numel()
            for p in model.parameters()
        )

        trainable = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        assert trainable == params

        grads = [
            p.grad
            for p in model.parameters()
            if p.requires_grad
        ]

        assert all(g is not None for g in grads)
        assert all(torch.isfinite(g).all() for g in grads)

        print("Parameters:", f"{params:,}")
        print("Trainable:", f"{trainable:,}")
        print("Loss:", round(loss.item(), 4))
        print("Forward pass: PASSED")
        print("Backward pass: PASSED")

        del model

    print("\nALL FOUR MODELS PASSED")


if __name__ == "__main__":
    main()
