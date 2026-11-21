
from pathlib import Path
import pandas as pd

PATH = Path("results/samples.csv")

if not PATH.exists():
    raise FileNotFoundError(
        "results/samples.csv not found. "
        "Run the existing dataset preparation first."
    )

df = pd.read_csv(PATH)

required = {"text", "aspect", "label", "image", "split", "group"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

print("\n===== DATASET AUDIT =====")

print("\nTotal samples:", len(df))

print("\nSplit distribution:")
print(df["split"].value_counts().to_string())

print("\nLabel distribution:")
print(
    pd.crosstab(df["split"], df["label"])
    .to_string()
)

print("\nMissing values:")
print(df[list(required)].isna().sum().to_string())

print("\nInvalid sentiment labels:")
print((~df["label"].isin([0, 1, 2])).sum())

print("\nSamples missing aspect placeholder:")
print(
    (~df["text"].str.contains(
        "[MASKED_ASPECT]",
        regex=False,
        na=False
    )).sum()
)

print("\nDuplicate group-aspect pairs:")
print(df.duplicated(["group", "aspect"]).sum())

print("\nGroups appearing in multiple splits:")
group_splits = df.groupby("group")["split"].nunique()
print((group_splits > 1).sum())

print("\nImage paths missing:")
missing_images = sum(
    not Path(str(path)).is_file()
    for path in df["image"]
)
print(missing_images)

print("\nExact masked-text duplicates:")
print(df.duplicated(["text"]).sum())

print("\nExact text appearing across splits:")
text_splits = df.groupby("text")["split"].nunique()
print((text_splits > 1).sum())

print("\nExact image appearing across splits:")
image_splits = df.groupby("image")["split"].nunique()
print((image_splits > 1).sum())

print("\nConflicting labels for same group and aspect:")
conflicts = (
    df.groupby(["group", "aspect"])["label"]
    .nunique()
)
print((conflicts > 1).sum())

print("\n===== AUDIT COMPLETE =====")

