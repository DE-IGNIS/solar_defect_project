
import copy
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from src.config import (
    DEVICE,
    TRAIN_CSV,
    VAL_CSV,
    WEIGHTS_DIR,
    METRICS_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_WORKERS,
    NUM_CLASSES,
    EXPERIMENTS,
    RANDOM_SEED,
)

from src.dataset import SolarThermalDataset
from src.model import build_model


def set_seed(seed=RANDOM_SEED):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_loaders(use_clahe):
    train_dataset = SolarThermalDataset(
        TRAIN_CSV,
        use_clahe=use_clahe,
        train=True,
    )

    val_dataset = SolarThermalDataset(
        VAL_CSV,
        use_clahe=use_clahe,
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    return train_loader, val_loader


def calculate_class_weights(dataset):
    counts = (
        dataset.data["y"]
        .value_counts()
        .sort_index()
        .values
    )

    counts = torch.tensor(counts, dtype=torch.float32)

    weights = torch.sqrt(
        counts.sum() / (NUM_CLASSES * counts.clamp(min=1))
    )

    return weights.to(DEVICE)


def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()

    running_loss = 0.0
    all_labels = []
    all_predictions = []

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=DEVICE.type,
            enabled=(DEVICE.type == "cuda"),
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_accuracy = accuracy_score(all_labels, all_predictions)
    epoch_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return epoch_loss, epoch_accuracy, epoch_f1


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()

    running_loss = 0.0
    all_labels = []
    all_predictions = []

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.amp.autocast(
            device_type=DEVICE.type,
            enabled=(DEVICE.type == "cuda"),
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_accuracy = accuracy_score(all_labels, all_predictions)
    epoch_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return epoch_loss, epoch_accuracy, epoch_f1


def run_experiment(use_clahe, tag):
    print(f"\n{'=' * 55}")
    print(f"STARTING EXPERIMENT: {tag.upper()}")
    print(f"CLAHE: {use_clahe}")
    print(f"Device: {DEVICE}")
    print(f"{'=' * 55}")

    train_loader, val_loader = create_loaders(use_clahe)

    model = build_model().to(DEVICE)

    class_weights = calculate_class_weights(
        train_loader.dataset
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(DEVICE.type == "cuda"),
    )

    best_f1 = -1.0
    best_state = None
    history = []

    for epoch in range(EPOCHS):

        train_loss, train_acc, train_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
        )

        val_loss, val_acc, val_f1 = validate(
            model,
            val_loader,
            criterion,
        )

        scheduler.step(val_f1)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Train F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "learning_rate": current_lr,
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())

            torch.save(
                best_state,
                WEIGHTS_DIR / f"thermal_{tag}.pth",
            )

            print("  -> Best checkpoint saved.")

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        METRICS_DIR / f"training_history_{tag}.csv",
        index=False,
    )

    print(f"\n{tag.upper()} COMPLETE")
    print(f"Best validation macro-F1: {best_f1:.4f}")

    return best_f1


def main():
    set_seed()

    results = {}

    for tag, use_clahe in EXPERIMENTS.items():
        results[tag] = run_experiment(
            use_clahe=use_clahe,
            tag=tag,
        )

    print("\nEXPERIMENT SUMMARY")
    for tag, score in results.items():
        print(f"{tag}: validation macro-F1 = {score:.4f}")


if __name__ == "__main__":
    main()
