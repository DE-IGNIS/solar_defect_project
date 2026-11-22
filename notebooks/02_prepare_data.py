
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split

from src.preprocess import preprocess_thermal, IMG_H, IMG_W

# Project paths
ROOT = Path.cwd()

DATA_ROOT = (
    ROOT / "data/thermal/InfraredSolarModules/"
    "dataset/InfraredSolarModules"
)

METADATA_PATH = DATA_ROOT / "module_metadata.json"

FIGURES_DIR = ROOT / "results/figures"
METRICS_DIR = ROOT / "results/metrics"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# Load metadata
with open(METADATA_PATH) as f:
    metadata = json.load(f)

df = pd.DataFrame.from_dict(
    metadata,
    orient="index"
).reset_index(names="id")

df["path"] = df["image_filepath"].apply(
    lambda p: str(DATA_ROOT / p)
)

df = df[["id", "path", "anomaly_class"]]

# Create consistent class IDs
classes = sorted(df["anomaly_class"].unique())

class_to_id = {
    name: index for index, name in enumerate(classes)
}

df["y"] = df["anomaly_class"].map(class_to_id)

with open(METRICS_DIR / "classes.json", "w") as f:
    json.dump(classes, f, indent=4)

# Stratified 70/15/15 split
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["y"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["y"],
    random_state=42
)

# Save splits
train_df.to_csv(METRICS_DIR / "train.csv", index=False)
val_df.to_csv(METRICS_DIR / "val.csv", index=False)
test_df.to_csv(METRICS_DIR / "test.csv", index=False)

# CLAHE before/after visualization
sample = df.sample(1, random_state=42).iloc[0]

original = cv2.imread(
    sample["path"],
    cv2.IMREAD_GRAYSCALE
)

enhanced = preprocess_thermal(
    original,
    use_clahe=True
)

# Convert normalized output back to display range
enhanced_display = (
    enhanced[0] * 0.229 + 0.485
) * 255

enhanced_display = enhanced_display.clip(0, 255).astype("uint8")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(original, cmap="inferno")
axes[0].set_title("Original Thermal Image")

axes[1].imshow(enhanced_display, cmap="inferno")
axes[1].set_title("CLAHE + Resize (64 × 96)")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "clahe_before_after.png",
    dpi=300
)

plt.close()

# Print report
print("PREPROCESSING COMPLETED")
print("-" * 40)

print("Classes:", len(classes))
print("Image dimensions:", IMG_H, "x", IMG_W)

print("\nDataset splits:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

print("\nSplit percentages:")
for name, split in [
    ("Train", train_df),
    ("Validation", val_df),
    ("Test", test_df)
]:
    print(f"{name}: {len(split) / len(df) * 100:.2f}%")

print("\nClass distribution verification:")
print(train_df["anomaly_class"].value_counts().sort_index())

print("\nFiles saved in:", METRICS_DIR)
print("CLAHE figure:", FIGURES_DIR / "clahe_before_after.png")
