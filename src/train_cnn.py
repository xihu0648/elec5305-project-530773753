
"""
ELEC5305 Project
CNN Training Pipeline

Model B: Time-resolved MFCC + CNN
Model C: Log-Mel Spectrogram + CNN

Modes:
    --self-test --representation mfcc
    --self-test --representation logmel
    --representation mfcc --fold 1
    --representation logmel --fold 1

Author: Xiao Hu
"""

from pathlib import Path
from config import load_config
import argparse
import copy
import json
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix
)

from train_cnn_architecture_test import AudioCNN
from cnn_dataset import (
    load_cache,
    MFCC_CACHE,
    LOGMEL_CACHE
)


# ==================================================
# Section 1: Configuration
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "cnn"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Shared experimental configuration
CONFIG = load_config()

SEED = CONFIG["project"]["random_seed"]
NUM_CLASSES = CONFIG["project"]["num_classes"]

BATCH_SIZE = CONFIG["training"]["batch_size"]
LEARNING_RATE = CONFIG["training"]["learning_rate"]
MAX_EPOCHS = CONFIG["training"]["max_epochs"]
PATIENCE = CONFIG["training"]["early_stopping_patience"]



def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==================================================
# Section 2: Synthetic Feature Generation
# ==================================================

def generate_synthetic_features(representation):
    """
    Generate artificial feature maps for testing.

    These are NOT UrbanSound8K features.
    """

    rng = np.random.default_rng(SEED)

    n_features = 40 if representation == "mfcc" else 128
    n_frames = 173

    samples_per_class_per_fold = 4

    X, y, folds = [], [], []

    frequency = np.arange(n_features, dtype=np.float32)

    for fold in range(1, 11):
        for class_id in range(NUM_CLASSES):

            centre = (
                (class_id + 0.5)
                * n_features / NUM_CLASSES
            )

            # Create a class-specific frequency pattern.
            pattern = np.exp(
                -0.5 * ((frequency - centre) / 2.5) ** 2
            )

            for _ in range(samples_per_class_per_fold):

                noise = rng.normal(
                    0, 0.15,
                    size=(n_features, n_frames)
                )

                feature = (
                    2.5 * pattern[:, None] + noise
                )

                X.append(feature.astype(np.float32))
                y.append(class_id)
                folds.append(fold)

    return (
        np.stack(X),
        np.asarray(y, dtype=np.int64),
        np.asarray(folds, dtype=np.int64)
    )


# ==================================================
# Section 3: Official Fold Splitting
# ==================================================

def prepare_data(X, y, folds, test_fold):

    validation_fold = (test_fold % 10) + 1

    train_mask = (
        (folds != test_fold)
        & (folds != validation_fold)
    )

    val_mask = folds == validation_fold
    test_mask = folds == test_fold

    assert train_mask.sum() > 0
    assert val_mask.sum() > 0
    assert test_mask.sum() > 0

    X_train = X[train_mask]
    X_val = X[val_mask]
    X_test = X[test_mask]

    y_train = y[train_mask]
    y_val = y[val_mask]
    y_test = y[test_mask]

    # Per-feature-row normalization.
    # Statistics are calculated ONLY from training data.
    mean = X_train.mean(
        axis=(0, 2),
        keepdims=True,
        dtype=np.float64
    ).astype(np.float32)

    std = X_train.std(
        axis=(0, 2),
        keepdims=True,
        dtype=np.float64
    ).astype(np.float32)

    std = np.maximum(std, 1e-6)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    print(f"\nTest fold: {test_fold}")
    print(f"Validation fold: {validation_fold}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Testing samples: {len(X_test)}")

    return (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        mean, std
    )


# ==================================================
# Section 4: PyTorch DataLoader
# ==================================================

def create_loader(X, y, shuffle=False):

    # CNN expects:
    # (batch, channels, feature_rows, time_frames)
    features = torch.from_numpy(
        np.ascontiguousarray(X[:, None], dtype=np.float32)
    )

    labels = torch.from_numpy(
        np.asarray(y, dtype=np.int64)
    )

    dataset = TensorDataset(features, labels)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )


# ==================================================
# Section 5: Train and Validate
# ==================================================

def run_epoch(model, loader, criterion, optimizer=None):

    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.set_grad_enabled(training):

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()

        total += batch_size

    return total_loss / total, correct / total


def train_model(train_loader, val_loader, max_epochs):

    model = AudioCNN().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    counter = 0

    for epoch in range(max_epochs):

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer
        )

        val_loss, val_acc = run_epoch(
            model, val_loader, criterion
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_loss - 1e-5:

            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            counter = 0

        else:

            counter += 1

            if counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_state)

    return model, history


# ==================================================
# Section 6: Evaluation
# ==================================================

def evaluate_model(model, test_loader):

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for inputs, labels in test_loader:

            inputs = inputs.to(DEVICE)

            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES))
    )

    return accuracy, macro_f1, cm


# ==================================================
# Section 7: Save Results
# ==================================================

def save_results(
    model, history, accuracy, macro_f1, cm,
    mean, std, representation, fold, self_test
):

    mode = "self_test" if self_test else "real"

    output_dir = (
        RESULTS_DIR / mode / representation / f"fold{fold}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "normalization_mean": mean,
            "normalization_std": std,
            "representation": representation,
            "fold": fold,
            "synthetic_test": self_test
        },
        output_dir / "model.pth"
    )

    metrics = {
        "representation": representation,
        "test_fold": fold,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "epochs": len(history["train_loss"]),
        "synthetic_test": self_test
    }

    with open(
        output_dir / "metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=4)

    np.save(output_dir / "confusion_matrix.npy", cm)

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(
        epochs, history["train_loss"],
        label="Training"
    )
    axes[0].plot(
        epochs, history["val_loss"],
        label="Validation"
    )
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        epochs, history["train_accuracy"],
        label="Training"
    )
    axes[1].plot(
        epochs, history["val_accuracy"],
        label="Validation"
    )
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        output_dir / "learning_curves.png",
        dpi=300
    )
    plt.close(fig)

    print(f"\nResults saved to: {output_dir}")


# ==================================================
# Section 8: Main
# ==================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--representation",
        choices=["mfcc", "logmel"],
        required=True
    )

    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        choices=range(1, 11)
    )

    parser.add_argument(
        "--self-test",
        action="store_true"
    )

    args = parser.parse_args()

    set_seed()

    print("=" * 55)
    print("ELEC5305 - CNN Training Pipeline")
    print("=" * 55)

    print(f"Device: {DEVICE}")
    print(f"Representation: {args.representation}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Max Epochs: {MAX_EPOCHS}")
    print(f"Early Stopping Patience: {PATIENCE}")

    if args.self_test:

        print("Mode: SYNTHETIC SELF-TEST")

        X, y, folds = generate_synthetic_features(
            args.representation
        )

        epochs = 5

    else:

        print("Mode: REAL URBANSOUND8K")

        cache_file = (
            MFCC_CACHE
            if args.representation == "mfcc"
            else LOGMEL_CACHE
        )

        X, y, folds = load_cache(cache_file)

        if len(X) != 8732:
            raise ValueError("Incomplete dataset cache.")

        np.testing.assert_array_equal(
            np.unique(folds),
            np.arange(1, 11)
        )

        epochs = MAX_EPOCHS

    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        mean, std
    ) = prepare_data(X, y, folds, args.fold)

    train_loader = create_loader(
        X_train, y_train, shuffle=True
    )

    val_loader = create_loader(X_val, y_val)
    test_loader = create_loader(X_test, y_test)

    model, history = train_model(
        train_loader, val_loader, epochs
    )

    accuracy, macro_f1, cm = evaluate_model(
        model, test_loader
    )

    print("\nTest Results")
    print("-" * 35)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    parameters = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print(f"Trainable parameters: {parameters:,}")

    save_results(
        model, history, accuracy, macro_f1, cm,
        mean, std,
        args.representation,
        args.fold,
        args.self_test
    )

    if args.self_test:
        print("\nSynthetic CNN self-test completed!")
        print("These are NOT UrbanSound8K results.")
    else:
        print("\nReal CNN fold experiment completed!")


if __name__ == "__main__":
    main()
