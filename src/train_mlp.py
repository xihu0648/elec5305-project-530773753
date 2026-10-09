
"""
ELEC5305 Project
Model A: Aggregated MFCC + MLP

Training, validation and evaluation pipeline.

Modes:
    --self-test : Test using synthetic features.
    --fold N    : Train on real UrbanSound8K data.

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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix
)

from dataset import load_cache


# ======================================================
# Section 1: Configuration
# ======================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_DIR / "results"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Load shared experimental configuration
CONFIG = load_config()

SEED = CONFIG["project"]["random_seed"]
INPUT_DIM = 80
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


# ======================================================
# Section 2: Define MLP
# ======================================================

class MFCC_MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(80, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.network(x)


# ======================================================
# Section 3: Prepare Fold Data
# ======================================================

def prepare_data(X, y, folds, test_fold):

    """
    The selected official fold is held out for testing.

    The validation fold is chosen deterministically
    from the remaining official folds.
    """

    if test_fold not in range(1, 11):
        raise ValueError("test_fold must be 1-10")

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

    assert not np.any(train_mask & test_mask)
    assert not np.any(val_mask & test_mask)

    X_train = X[train_mask]
    X_val = X[val_mask]
    X_test = X[test_mask]

    y_train = y[train_mask]
    y_val = y[val_mask]
    y_test = y[test_mask]

    # Fit scaler ONLY on training samples.
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"\nTest fold: {test_fold}")
    print(f"Validation fold: {validation_fold}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Testing samples: {len(X_test)}")

    return (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        scaler
    )


# ======================================================
# Section 4: Create DataLoaders
# ======================================================

def make_loader(X, y, shuffle=False):

    features = torch.tensor(
        X, dtype=torch.float32
    )

    labels = torch.tensor(
        y, dtype=torch.long
    )

    dataset = TensorDataset(features, labels)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle
    )

    return loader


# ======================================================
# Section 5: Training
# ======================================================

def train_model(train_loader, val_loader):

    model = MFCC_MLP().to(DEVICE)

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

    best_val_loss = float("inf")
    best_state = None

    patience_counter = 0

    for epoch in range(MAX_EPOCHS):

        # -------------------------
        # Training phase
        # -------------------------

        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += len(labels)

        train_loss = total_loss / total
        train_accuracy = correct / total

        # -------------------------
        # Validation phase
        # -------------------------

        model.eval()

        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for inputs, labels in val_loader:

                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                val_loss_sum += (
                    loss.item() * len(labels)
                )

                predictions = outputs.argmax(dim=1)

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += len(labels)

        val_loss = val_loss_sum / val_total
        val_accuracy = val_correct / val_total

        # Save training history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        history["train_accuracy"].append(
            train_accuracy
        )

        history["val_accuracy"].append(
            val_accuracy
        )

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        # -------------------------
        # Early stopping
        # -------------------------

        if val_loss < best_val_loss - 1e-5:

            best_val_loss = val_loss

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print("\nEarly stopping triggered.")
                break

    # Restore best validation model
    model.load_state_dict(best_state)

    return model, history


# ======================================================
# Section 6: Model Evaluation
# ======================================================

def evaluate_model(model, test_loader):

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for inputs, labels in test_loader:

            inputs = inputs.to(DEVICE)

            outputs = model(inputs)

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    accuracy = accuracy_score(
        all_labels, all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        labels=list(range(NUM_CLASSES)),
        zero_division=0
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES))
    )

    return accuracy, macro_f1, cm


# ======================================================
# Section 7: Save Results
# ======================================================

def save_results(
    model,
    scaler,
    history,
    accuracy,
    macro_f1,
    cm,
    test_fold,
    self_test=False
):

    if self_test:
        result_dir = RESULTS_DIR / "self_test"
    else:
        result_dir = RESULTS_DIR / "mlp" / f"fold{test_fold}"

    result_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save model and scaler parameters
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "test_fold": test_fold
        },
        result_dir / "model.pth"
    )

    # Save numerical results
    metrics = {
        "test_fold": int(test_fold),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "epochs_completed": len(
            history["train_loss"]
        ),
        "synthetic_test": self_test
    }

    with open(
        result_dir / "metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(metrics, file, indent=4)

    np.save(
        result_dir / "confusion_matrix.npy",
        cm
    )

    # Save learning curves
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4)
    )

    epochs = np.arange(
        1, len(history["train_loss"]) + 1
    )

    axes[0].plot(
        epochs,
        history["train_loss"],
        label="Training Loss"
    )

    axes[0].plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        epochs,
        history["train_accuracy"],
        label="Training Accuracy"
    )

    axes[1].plot(
        epochs,
        history["val_accuracy"],
        label="Validation Accuracy"
    )

    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        result_dir / "learning_curves.png",
        dpi=300
    )

    plt.close(fig)

    print(f"\nResults saved to: {result_dir}")


# ======================================================
# Section 8: Synthetic Data
# ======================================================

def generate_synthetic_data():

    """
    Generate learnable synthetic features.

    Each class has a different feature centre.
    This is only a software functionality test.
    """

    rng = np.random.default_rng(SEED)

    X_list = []
    y_list = []
    fold_list = []

    class_centres = rng.normal(
        0, 2, size=(NUM_CLASSES, INPUT_DIM)
    )

    for fold in range(1, 11):

        for class_id in range(NUM_CLASSES):

            features = (
                class_centres[class_id]
                + rng.normal(
                    0, 0.6, size=(20, INPUT_DIM)
                )
            )

            X_list.append(features)

            y_list.extend([class_id] * 20)
            fold_list.extend([fold] * 20)

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    folds = np.array(fold_list, dtype=np.int64)

    return X, y, folds


# ======================================================
# Section 9: Main Program
# ======================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--self-test",
        action="store_true"
    )

    parser.add_argument(
        "--fold",
        type=int,
        default=1
    )

    args = parser.parse_args()

    set_seed()

    print("=" * 55)
    print("ELEC5305 - MFCC MLP Training Pipeline")
    print("=" * 55)

    print(f"\nDevice: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Max Epochs: {MAX_EPOCHS}")
    print(f"Early Stopping Patience: {PATIENCE}")

    if args.self_test:

        print("Mode: SYNTHETIC SELF-TEST")

        X, y, folds = generate_synthetic_data()

    else:

        print("Mode: REAL URBANSOUND8K")

        X, y, folds = load_cache()

        if len(X) != 8732:
            raise ValueError(
                "Incomplete UrbanSound8K feature cache"
            )

        if not np.array_equal(
            np.unique(folds),
            np.arange(1, 11)
        ):
            raise ValueError("Official folds missing")

    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        scaler
    ) = prepare_data(
        X, y, folds, args.fold
    )

    train_loader = make_loader(
        X_train, y_train, shuffle=True
    )

    val_loader = make_loader(
        X_val, y_val
    )

    test_loader = make_loader(
        X_test, y_test
    )

    model, history = train_model(
        train_loader,
        val_loader
    )

    accuracy, macro_f1, cm = evaluate_model(
        model,
        test_loader
    )

    print("\nTest Results")
    print("-" * 35)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    num_parameters = sum(
        p.numel() for p in model.parameters()
    )

    print(f"Model parameters: {num_parameters:,}")

    save_results(
        model,
        scaler,
        history,
        accuracy,
        macro_f1,
        cm,
        args.fold,
        args.self_test
    )

    if args.self_test:
        print("\nSynthetic training self-test completed!")
        print("These metrics are NOT UrbanSound8K results.")
    else:
        print("\nReal fold experiment completed!")


if __name__ == "__main__":
    main()
