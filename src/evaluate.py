
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.config import (
    DEVICE,
    TEST_CSV,
    CLASSES_JSON,
    WEIGHTS_DIR,
    METRICS_DIR,
    FIGURES_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    NUM_CLASSES,
    EXPERIMENTS,
)

from src.dataset import SolarThermalDataset
from src.model import build_model


def load_classes():
    with open(CLASSES_JSON, "r") as f:
        data = json.load(f)

    # Supports either a list or a class-to-index dictionary.
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        return [
            name for name, index in
            sorted(data.items(), key=lambda item: item[1])
        ]

    raise ValueError("Unsupported classes.json format")


@torch.no_grad()
def get_predictions(model, loader):
    model.eval()

    all_labels = []
    all_predictions = []

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)

        with torch.amp.autocast(
            device_type=DEVICE.type,
            enabled=(DEVICE.type == "cuda"),
        ):
            outputs = model(images)

        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.numpy().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    return np.array(all_labels), np.array(all_predictions)


def evaluate_experiment(tag, use_clahe, classes):
    print(f"\nEvaluating: {tag.upper()}")

    dataset = SolarThermalDataset(
        TEST_CSV,
        use_clahe=use_clahe,
        train=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = build_model(pretrained=False).to(DEVICE)

    checkpoint_path = WEIGHTS_DIR / f"thermal_{tag}.pth"

    state = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state)

    y_true, y_pred = get_predictions(model, loader)

    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=classes,
        digits=4,
        zero_division=0,
    )

    normal_id = classes.index("No-Anomaly")

    binary_true = (y_true != normal_id).astype(int)
    binary_pred = (y_pred != normal_id).astype(int)

    binary_f1 = f1_score(
        binary_true,
        binary_pred,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        normalize="true",
    )

    # Save normalized confusion matrix.
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar()

    plt.xticks(range(NUM_CLASSES), classes, rotation=45, ha="right")
    plt.yticks(range(NUM_CLASSES), classes)

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Normalized Confusion Matrix - {tag.upper()}")

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            plt.text(
                j, i, f"{cm[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )

    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / f"confusion_matrix_{tag}.png",
        dpi=200,
    )
    plt.close()

    # Save text report.
    with open(METRICS_DIR / f"classification_report_{tag}.txt", "w") as f:
        f.write(report)
        f.write(f"\nAccuracy: {accuracy:.4f}\n")
        f.write(f"Macro-F1: {macro_f1:.4f}\n")
        f.write(f"Binary anomaly F1: {binary_f1:.4f}\n")

    results = {
        "experiment": tag,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "binary_anomaly_f1": float(binary_f1),
    }

    print(report)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Binary anomaly F1: {binary_f1:.4f}")

    return results


def main():
    classes = load_classes()
    all_results = []

    for tag, use_clahe in EXPERIMENTS.items():
        result = evaluate_experiment(
            tag,
            use_clahe,
            classes,
        )
        all_results.append(result)

    pd.DataFrame(all_results).to_csv(
        METRICS_DIR / "ablation_results.csv",
        index=False,
    )

    print("\nFINAL ABLATION RESULTS")
    print(pd.DataFrame(all_results).to_string(index=False))


if __name__ == "__main__":
    main()
