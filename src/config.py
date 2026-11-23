
from pathlib import Path
import torch

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "thermal" / "InfraredSolarModules" / "dataset" / "InfraredSolarModules"
IMAGE_DIR = DATA_DIR / "images"

RESULTS_DIR = PROJECT_ROOT / "results"
WEIGHTS_DIR = RESULTS_DIR / "weights"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

TRAIN_CSV = METRICS_DIR / "train.csv"
VAL_CSV = METRICS_DIR / "val.csv"
TEST_CSV = METRICS_DIR / "test.csv"
CLASSES_JSON = METRICS_DIR / "classes.json"

# --------------------------------------------------
# IMAGE CONFIGURATION
# --------------------------------------------------

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 96
IMAGE_CHANNELS = 3

USE_IMAGENET_NORMALIZATION = True

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------

MODEL_NAME = "resnet18"
NUM_CLASSES = 12
PRETRAINED = True

# --------------------------------------------------
# TRAINING CONFIGURATION
# --------------------------------------------------

BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 2
RANDOM_SEED = 42

# --------------------------------------------------
# DEVICE
# --------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"

# --------------------------------------------------
# EXPERIMENTS
# --------------------------------------------------

EXPERIMENTS = {
    "raw": False,
    "clahe": True,
}

# --------------------------------------------------
# OUTPUT DIRECTORIES
# --------------------------------------------------

for directory in (WEIGHTS_DIR, METRICS_DIR, FIGURES_DIR):
    directory.mkdir(parents=True, exist_ok=True)

print(f"Project: {PROJECT_ROOT}")
print(f"Device: {DEVICE}")
