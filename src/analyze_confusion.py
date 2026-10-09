
"""
ELEC5305 Project
Official 10-Fold Confusion Matrix Analysis

Combines the confusion matrices from all ten
official UrbanSound8K test folds.

Author: Xiao Hu
"""

from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay


PROJECT_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_DIR / "results"

OUTPUT_DIR = RESULTS_DIR / "confusion_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


CLASS_NAMES = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]

MODELS = {
    "MFCC-MLP": RESULTS_DIR / "mlp",
    "MFCC-CNN": RESULTS_DIR / "cnn" / "real" / "mfcc",
    "Log-Mel CNN": RESULTS_DIR / "cnn" / "real" / "logmel"
}


def load_confusion_matrices(model_dir):

    total_cm = np.zeros((10, 10), dtype=np.int64)

    for fold in range(1, 11):

        fold_dir = model_dir / f"fold{fold}"

        cm_file = fold_dir / "confusion_matrix.npy"
        metrics_file = fold_dir / "metrics.json"

        if not cm_file.exists():
            raise FileNotFoundError(cm_file)

        if not metrics_file.exists():
            raise FileNotFoundError(metrics_file)

        with open(metrics_file, encoding="utf-8") as f:
            metrics = json.load(f)

        assert metrics["test_fold"] == fold
        assert metrics["synthetic_test"] is False

        cm = np.load(cm_file, allow_pickle=False)

        assert cm.shape == (10, 10)
        assert np.issubdtype(cm.dtype, np.integer)
        assert np.all(cm >= 0)

        total_cm += cm.astype(np.int64)

    assert total_cm.sum() == 8732, (
        "The combined confusion matrix must "
        "contain exactly 8732 recordings."
    )

    return total_cm


def plot_confusion_matrix(cm, model_name, filename):

    # Row-normalized confusion matrix.
    # Each row corresponds to one true class.
    row_sums = cm.sum(axis=1, keepdims=True)

    normalized = np.divide(
        cm.astype(float),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0
    )

    fig, ax = plt.subplots(figsize=(11, 9))

    display = ConfusionMatrixDisplay(
        confusion_matrix=normalized,
        display_labels=CLASS_NAMES
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format=".2f",
        xticks_rotation=45,
        colorbar=True
    )

    ax.set_title(
        f"{model_name} - Official 10-Fold Confusion Matrix"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def analyze_model(model_name, model_dir):

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    cm = load_confusion_matrices(model_dir)

    np.save(
        OUTPUT_DIR / (
            model_name.lower()
            .replace("-", "_")
            .replace(" ", "_")
            + "_confusion_matrix.npy"
        ),
        cm
    )

    filename = (
        model_name.lower()
        .replace("-", "_")
        .replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plot_confusion_matrix(cm, model_name, filename)

    # Per-class recall
    support = cm.sum(axis=1)

    recall = np.divide(
        np.diag(cm),
        support,
        out=np.zeros(10, dtype=float),
        where=support != 0
    )

    print("\nPer-class Recall")

    for name, value in zip(CLASS_NAMES, recall):
        print(f"{name:<20}: {value:.4f}")

    # Extract the most frequent directional errors
    errors = cm.copy()
    np.fill_diagonal(errors, 0)

    pairs = []

    for true_class in range(10):
        for predicted_class in range(10):

            if true_class == predicted_class:
                continue

            pairs.append((
                int(errors[true_class, predicted_class]),
                CLASS_NAMES[true_class],
                CLASS_NAMES[predicted_class]
            ))

    pairs.sort(reverse=True)

    print("\nTop 10 Classification Confusions")
    print("True Class -> Predicted Class")

    for count, true_name, pred_name in pairs[:10]:

        print(
            f"{true_name:<20} -> "
            f"{pred_name:<20}: {count}"
        )

    # Specific drilling vs jackhammer analysis
    drilling_id = CLASS_NAMES.index("drilling")
    jackhammer_id = CLASS_NAMES.index("jackhammer")

    print("\nDrilling / Jackhammer Analysis")

    print(
        "Drilling predicted as Jackhammer:",
        cm[drilling_id, jackhammer_id]
    )

    print(
        "Jackhammer predicted as Drilling:",
        cm[jackhammer_id, drilling_id]
    )

    print(f"\nFigure saved: {OUTPUT_DIR / filename}")


def main():

    print("=" * 60)
    print("ELEC5305 - Official 10-Fold Confusion Analysis")
    print("=" * 60)

    for model_name, model_dir in MODELS.items():
        analyze_model(model_name, model_dir)

    print("\nAll confusion matrices processed successfully!")


if __name__ == "__main__":
    main()
