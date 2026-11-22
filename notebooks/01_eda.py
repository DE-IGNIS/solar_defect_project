import json
import random
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Reproducibility
random.seed(42)

# Paths
PROJECT_ROOT = Path.cwd()
DATA_ROOT = (
    PROJECT_ROOT
    / "data/thermal/InfraredSolarModules/dataset/InfraredSolarModules"
)

IMAGE_DIR = DATA_ROOT / "images"
METADATA_PATH = DATA_ROOT / "module_metadata.json"
OUTPUT_DIR = PROJECT_ROOT / "results/figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

# 1. Class distribution
class_counts = df["anomaly_class"].value_counts().sort_values(
    ascending=False
)

plt.figure(figsize=(13, 6))

class_counts.plot(
    kind="bar",
    edgecolor="black"
)

plt.title("Thermal Dataset Class Distribution")
plt.xlabel("Defect Class")
plt.ylabel("Number of Images")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "class_dist.png",
    dpi=300
)

plt.close()

# 2. Sample image grid
classes = sorted(df["anomaly_class"].unique())

fig, axes = plt.subplots(
    12, 3,
    figsize=(9, 24)
)

for row, label in enumerate(classes):

    samples = df[df["anomaly_class"] == label]

    selected = samples.sample(
        n=min(3, len(samples)),
        random_state=42
    )

    for col in range(3):

        ax = axes[row, col]

        if col < len(selected):

            image_path = selected.iloc[col]["path"]

            image = Image.open(image_path).convert("L")

            ax.imshow(
                image,
                cmap="inferno",
                interpolation="nearest"
            )

            ax.set_title(label if col == 0 else "")

        ax.axis("off")

fig.suptitle(
    "Representative Thermal Images by Class",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "samples.png",
    dpi=300
)

plt.close()

print("EDA completed successfully.")
print("Class distribution:", OUTPUT_DIR / "class_dist.png")
print("Sample grid:", OUTPUT_DIR / "samples.png")