# ImplicitMABSA: Masked-Aspect Multimodal Sentiment Analysis

A reproducible **masked-aspect multimodal sentiment classification** project using text and images.

The project replaces annotated target expressions with `[MASKED_ASPECT]`, retains their original sentiment labels, and compares four deep learning architectures using frozen pretrained encoders and trainable classification/fusion heads.

**Important:** This task is a proxy for implicit aspect understanding, not a naturally implicit-aspect dataset. The target identity is hidden, but its position remains visible. Some tweets contain other mentions of the same entity, and some labels become ambiguous after masking.

## 1. Project objectives

- Investigate sentiment classification when explicit target expressions are masked.
- Compare text-only, image-only, and multimodal architectures.
- Explore bidirectional cross-attention between textual tokens and visual patches.
- Evaluate models using accuracy, precision, recall, and macro-F1.
- Provide an interactive Gradio demonstration.

## 2. Dataset

**Source:** [TwitterDataMABSA](https://github.com/CopotronicRifat/TwitterDataMABSA)

The project combines Twitter-2015 and Twitter-2017 multimodal aspect-based sentiment datasets.

### Download

```bash
git clone https://github.com/CopotronicRifat/TwitterDataMABSA.git data/TwitterDataMABSA
```

Expected directory structure:

```text
data/TwitterDataMABSA/
    twitter2015/
        train.txt
        dev.txt
        test.txt
    twitter2017/
        train.txt
        dev.txt
        test.txt
    twitter2015_images/
    twitter2017_images/
```

If image folders are missing, consult the dataset resources linked by [CMMT-Code](https://github.com/yangli-hub/CMMT-Code).

The image archives must be extracted into the expected directories before training.

### Dataset preprocessing

Each original record contains four lines:

1. Text template containing `$T$`
2. Target aspect
3. Sentiment polarity (-1, 0, or 1)
4. Image filename

The preprocessing pipeline:

- Loads Twitter-2015 and Twitter-2017.
- Verifies the associated image exists.
- Replaces `$T$` with `[MASKED_ASPECT]`.
- Converts labels to 0 = negative, 1 = neutral, 2 = positive.
- Filters repeated group-aspect records.
- Prevents exact constructed tweet groups from appearing across multiple splits.
- Creates a group-based validation split if none exists.

### Full-dataset distribution

| Split | Samples |
|---|---:|
| Training | 6,680 |
| Validation | 2,274 |
| Testing | 2,259 |
| **Total** | **11,213** |

### Class distribution

| Split | Negative | Neutral | Positive |
|---|---:|---:|---:|
| Training | 776 | 3,492 | 2,412 |
| Validation | 290 | 1,179 | 805 |
| Testing | 279 | 1,173 | 807 |

## 3. Model architectures

The project implements four models.

### Model 1 — RoBERTa text-only

- Pretrained RoBERTa-base encoder.
- Frozen encoder parameters.
- First-token representation with 768 features.
- Trainable MLP classifier.
- Three sentiment classes.

### Model 2 — ViT image-only

- Pretrained ViT-base-patch16-224 encoder.
- Frozen encoder parameters.
- First-token visual representation with 768 features.
- Trainable MLP classifier.
- Three sentiment classes.

### Model 3 — CLIP concatenation

- Pretrained CLIP ViT-B/32.
- Frozen text and image encoders.
- First projected text-token representation (512 dimensions).
- First projected visual-token representation (512 dimensions).
- Concatenated representation (1,024 dimensions).
- Trainable MLP classifier.

This baseline uses the first text token rather than CLIP's conventional end-of-text pooled embedding.

### Model 4 — CLIP bidirectional cross-attention

The proposed fusion architecture uses projected token-level and patch-level CLIP representations.

**Text-to-image attention:**
- Text representations act as queries.
- Visual representations act as keys and values.

**Image-to-text attention:**
- Visual representations act as queries.
- Text representations act as keys and values.
- Text padding is masked in this attention direction.

Both attention modules use eight attention heads.

The architecture then applies residual connections, layer normalization, masked mean pooling for text, mean pooling for images, feature concatenation, and an MLP classifier.

**All pretrained encoders remain frozen. Only classification and fusion components are trained.**

## 4. Installation

Python 3.10 or 3.11 is recommended.

Create and activate a virtual environment.

On Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

For CUDA acceleration, install a compatible PyTorch build using the instructions at [PyTorch](https://pytorch.org/get-started/locally/).

## 5. Training

### Smoke-test experiment

```powershell
python -m src.train --limit 500 --epochs 3 --batch 16 --extract-batch 4
```

### Full-dataset V2 experiment

The following configuration produced the reported V2 results:

```powershell
python -m src.train --limit 0 --epochs 5 --batch 16 --extract-batch 4
```

### Training configuration

| Parameter | Value |
|---|---|
| Random seed | 42 |
| Maximum epochs | 5 |
| Classification batch size | 16 |
| Feature-extraction batch size | 4 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.01 |
| Loss | Class-weighted cross-entropy |
| Scheduler | Linear with 10% warmup |
| Gradient clipping | 1.0 |
| Early stopping patience | 2 |
| Checkpoint selection | Best validation macro-F1 |

Mixed precision is used when CUDA is available.

To rebuild cached features:

```powershell
python -m src.train --limit 0 --epochs 5 --batch 16 --extract-batch 4 --rebuild-cache
```

**Note:** Running training again may overwrite existing checkpoints and result files. Back up a completed experiment before retraining.

## 6. Model evaluation

Run:

```powershell
python -m src.evaluate
```

### Full-dataset V2 test results

| Model | Accuracy | Macro precision | Macro recall | Macro-F1 |
|---|---:|---:|---:|---:|
| RoBERTa text-only | 0.6131 | 0.5752 | 0.6159 | 0.5886 |
| ViT image-only | 0.4776 | 0.4448 | 0.4703 | 0.4486 |
| CLIP concatenation | 0.4976 | 0.4852 | 0.5182 | 0.4830 |
| **CLIP cross-attention** | **0.6445** | **0.6161** | **0.6545** | **0.6310** |

### Key findings

CLIP bidirectional cross-attention achieved the highest accuracy and macro-F1 among the four evaluated models.

Compared with RoBERTa text-only, cross-attention improved:

- Accuracy by approximately 3.14 percentage points.
- Macro-F1 by approximately 4.24 percentage points.

These improvements are observed experimental differences; statistical significance has not been established.

## 7. Dataset integrity audit

Run:

```powershell
python -m src.audit_dataset
```

The full-dataset V2 audit reported:

| Check | Result |
|---|---:|
| Total samples | 11,213 |
| Missing values | 0 |
| Invalid sentiment labels | 0 |
| Missing masked-aspect placeholders | 0 |
| Duplicate group-aspect pairs | 0 |
| Groups appearing in multiple splits | 0 |
| Missing image files | 0 |
| Exact image paths across splits | 0 |
| Exact masked-text duplicate rows | 26 |
| Unique masked texts across splits | 7 |

The seven repeated masked-text values affected 27 records. Manual inspection showed different aspect identities and image paths.

These observations do not establish that near-duplicate images or semantically overlapping tweets are absent.

## 8. Interactive Gradio application

Launch the application:

```powershell
python -m src.app
```

Open:

```text
http://127.0.0.1:7860
```

The application accepts text and an image and displays sentiment predictions from all four trained models.

Model confidence scores are softmax outputs and should not be interpreted as calibrated probabilities.

## 9. Generated outputs

Training and evaluation produce files such as:

```text
checkpoints/
    text.pt
    image.pt
    concat.pt
    cross.pt

results/
    samples.csv
    history.csv
    comparison.csv
    comparison.png
    cross_correct_text_wrong.csv
```

Additional confusion matrices and training curves are generated by evaluation.

Cached features, datasets, and model checkpoints are excluded from GitHub using `.gitignore`.

## 10. MVSA-Single fallback

MVSA-Single contains tweet-image pairs with overall sentiment annotations rather than target-specific aspect sentiment annotations.

It is not a direct replacement for Twitter-2015/2017 in this experiment.

To use MVSA-Single for aspect-based sentiment analysis, target aspects and corresponding sentiment labels would need to be annotated separately.

Otherwise, the task should be described as multimodal overall sentiment classification.

## 11. Limitations

1. **Synthetic aspect masking:** The dataset does not contain naturally implicit aspect annotations.
2. **Aspect-position information:** The `[MASKED_ASPECT]` placeholder preserves the original aspect position.
3. **Ambiguous examples:** Some sentences become nearly empty or contain only the placeholder.
4. **Class imbalance:** Neutral examples are more frequent than other classes.
5. **Frozen encoders:** Pretrained backbone representations are not fine-tuned for the task.
6. **Concatenation baseline:** The first CLIP text token is used instead of its standard pooled text embedding.
7. **Potential semantic duplication:** Exact group overlaps were filtered, but near-duplicate text and image content have not been exhaustively audited.
8. **Modified benchmark splits:** Filtering overlapping groups changes the original benchmark composition.
9. **Statistical uncertainty:** Improvements have not been validated using repeated independent runs or significance testing.

## 12. Reproducibility

The project uses random seed 42 and saves model checkpoints based on validation macro-F1.

Dataset preparation, feature extraction, training, evaluation, and demonstration are implemented as separate Python modules.

For systems with limited GPU memory, reduce `--extract-batch` to 2 or 4.

Pretrained model downloads require internet access and sufficient disk space.

## 13. References

- [TwitterDataMABSA dataset](https://github.com/CopotronicRifat/TwitterDataMABSA)
- [CMMT-Code](https://github.com/yangli-hub/CMMT-Code)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PyTorch](https://pytorch.org/)