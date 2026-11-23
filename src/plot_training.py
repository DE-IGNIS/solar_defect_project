
import pandas as pd
import matplotlib.pyplot as plt

from src.config import METRICS_DIR, FIGURES_DIR


def plot_metric(metric, title, filename):
    plt.figure(figsize=(9, 5))

    for tag in ["raw", "clahe"]:
        path = METRICS_DIR / f"training_history_{tag}.csv"
        df = pd.read_csv(path)

        plt.plot(
            df["epoch"],
            df[metric],
            marker="o",
            label=tag.upper(),
        )

    plt.xlabel("Epoch")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / filename, dpi=200)
    plt.close()


def main():
    plot_metric(
        "train_loss",
        "Training Loss",
        "training_loss.png",
    )

    plot_metric(
        "val_loss",
        "Validation Loss",
        "validation_loss.png",
    )

    plot_metric(
        "val_macro_f1",
        "Validation Macro-F1",
        "validation_macro_f1.png",
    )

    print("Training plots saved in:", FIGURES_DIR)


if __name__ == "__main__":
    main()
