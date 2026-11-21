
import torch
from torch import nn


class TextEncoder(nn.Module):
    """Randomly initialized embedding + bidirectional LSTM."""

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)

        # Padding positions do not participate in LSTM recurrence.
        lengths = attention_mask.long().sum(dim=1).clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_output, _ = self.lstm(packed)

        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=input_ids.size(1)
        )

        return output


class ImageEncoder(nn.Module):
    """Lightweight CNN trained from random initialization."""

    def __init__(self, feature_dim=256):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),

            # 128x128 input -> 8x8 spatial features.
            nn.AdaptiveAvgPool2d((8, 8))
        )

    def forward(self, images):
        features = self.features(images)
        batch, channels, height, width = features.shape

        return features.flatten(2).transpose(1, 2)


def masked_mean(x, mask):
    mask = mask.unsqueeze(-1).to(x.dtype)

    return (
        (x * mask).sum(dim=1)
        / mask.sum(dim=1).clamp(min=1)
    )


class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class BiLSTMTextOnly(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.text_encoder = TextEncoder(vocab_size)
        self.classifier = ClassificationHead(256)

    def forward(self, batch):
        text_features = self.text_encoder(
            batch["input_ids"],
            batch["attention_mask"]
        )

        pooled = masked_mean(
            text_features,
            batch["attention_mask"]
        )

        return self.classifier(pooled)


class CNNImageOnly(nn.Module):
    def __init__(self, vocab_size=None):
        super().__init__()

        self.image_encoder = ImageEncoder()
        self.classifier = ClassificationHead(256)

    def forward(self, batch):
        image_features = self.image_encoder(batch["image"])

        pooled = image_features.mean(dim=1)

        return self.classifier(pooled)


class ConcatFusion(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.text_encoder = TextEncoder(vocab_size)
        self.image_encoder = ImageEncoder()

        self.classifier = ClassificationHead(512)

    def forward(self, batch):
        text_features = self.text_encoder(
            batch["input_ids"],
            batch["attention_mask"]
        )

        image_features = self.image_encoder(batch["image"])

        text_pooled = masked_mean(
            text_features,
            batch["attention_mask"]
        )

        image_pooled = image_features.mean(dim=1)

        fused = torch.cat(
            [text_pooled, image_pooled],
            dim=-1
        )

        return self.classifier(fused)


class CrossAttentionFusion(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.text_encoder = TextEncoder(vocab_size)
        self.image_encoder = ImageEncoder()

        self.text_to_image = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        self.image_to_text = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        self.text_norm = nn.LayerNorm(256)
        self.image_norm = nn.LayerNorm(256)

        self.classifier = ClassificationHead(512)

    def forward(self, batch):
        text = self.text_encoder(
            batch["input_ids"],
            batch["attention_mask"]
        )

        image = self.image_encoder(batch["image"])

        mask = batch["attention_mask"].bool()

        # Text queries attend to image spatial features.
        text_attended, _ = self.text_to_image(
            query=text,
            key=image,
            value=image,
            need_weights=False
        )

        # Image queries attend to valid text tokens.
        image_attended, _ = self.image_to_text(
            query=image,
            key=text,
            value=text,
            key_padding_mask=~mask,
            need_weights=False
        )

        text = self.text_norm(text + text_attended)
        image = self.image_norm(image + image_attended)

        text_pooled = masked_mean(text, mask)
        image_pooled = image.mean(dim=1)

        fused = torch.cat(
            [text_pooled, image_pooled],
            dim=-1
        )

        return self.classifier(fused)


MODELS = {
    "bilstm": BiLSTMTextOnly,
    "cnn": CNNImageOnly,
    "concat": ConcatFusion,
    "cross": CrossAttentionFusion
}
