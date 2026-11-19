# ImplicitMABSA: Masked-Aspect Multimodal Sentiment

A reproducible *masked-aspect* sentiment benchmark: replace annotated target expressions with `[MASKED_ASPECT]`, retain their original sentiment labels, and compare frozen pretrained feature extractors with trained classification/fusion heads.

**Important:** This is a proxy for implicit aspect understanding, not a naturally implicit-aspect dataset. The target is hidden from the model. Some tweets contain other explicit mentions of the same entity, and some labels become ambiguous after masking. Inspect examples before making research claims.

## Data source

https://github.com/CopotronicRifat/TwitterDataMABSA

Download from a terminal with internet:

```bash
git clone https://github.com/CopotronicRifat/TwitterDataMABSA.git data/TwitterDataMABSA
```

Expected files (dataset repository provides `twitter2015/`, `twitter2017/`, `twitter2015_images/`, `twitter2017_images/`):

```
data/TwitterDataMABSA/
  twitter2015/train.txt
  twitter2015/dev.txt
  twitter2015/test.txt
  twitter2017/train.txt
  twitter2017/dev.txt
  twitter2017/test.txt
  twitter2015_images/*.jpg
  twitter2017_images/*.jpg
```

Check exact names after cloning. If image folders are missing, download the image archive from the CMMT repository's linked Google Drive and extract by year into the two image directories: https://github.com/yangli-hub/CMMT-Code . Missing image pairs are skipped and their counts printed. Do not assume the git repository contains all images until checked.

Each Twitter record is four lines: template with `$T$`, target aspect, polarity (-1/0/1), image filename. Split boundaries are respected; when no dev split exists, a group-based 15% validation partition is carved from train. Repeated tweet groups are filtered to avoid overlap.

## Install and run

Python 3.10 or 3.11 recommended. For NVIDIA T4, install GPU PyTorch using https://pytorch.org/get-started/locally/ first, then:

```bash
pip install -r requirements.txt
python -m src.train --limit 500 --epochs 3 --batch 32 --extract-batch 16
python -m src.evaluate
python -m src.app
```

Full data:

```bash
python -m src.train --limit 0 --epochs 5 --batch 32 --extract-batch 16
python -m src.evaluate
python -m src.app
```

To reset feature caches for the current data, add `--rebuild-cache`. Cache files are keyed by the masked sample table hash. Features are computed once using frozen RoBERTa-base, ViT-base, and CLIP ViT-B/32, then four small heads are trained with class weights, AdamW, warmup, mixed precision on CUDA, validation macro-F1 early stopping and checkpoint saving. **The pretrained backbones are NOT fine-tuned.**

Output: `results/samples.csv`, `history.csv`, `comparison.csv`, confusion matrices, training curves, comparison chart, `cross_correct_text_wrong.csv`; checkpoints in `checkpoints/`.

## MVSA-Single fallback

MVSA-Single (search for the official MVSA dataset release / published mirrors) contains tweet-image pairs with *overall sentiment*, not annotated target aspects. It is **not a drop-in replacement** for aspect-specific Twitter-2015/2017. To use it, first manually annotate a target aspect and its sentiment for each example, then export the same four-line Twitter format and image folder. Otherwise label the experiment *multimodal overall sentiment*, not ImplicitMABSA. For a deadline, prefer repairing the Twitter image download or using a smaller valid subset.

## Reproducibility and limitations

Seed 42. Validation is used for model selection; test is evaluated once. Image-only does not know which aspect was masked, so multi-aspect examples are inherently ambiguous. Cross-attention uses CLIP token and patch representations; concatenation uses their first-token representations. Model confidence is softmax, not calibrated probability. Pretrained model downloads require internet and disk space. T4 strongly recommended; 4 GB GPU may OOM on CLIP feature extraction—reduce `--extract-batch` to 2 or 4. Cache and checkpoint files are not committed.
