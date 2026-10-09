
"""
ELEC5305 Project
Model Performance Metrics

Evaluate:
1. Accuracy
2. Macro-F1
3. Per-class Precision and Recall
4. Confusion Matrix
5. Model Parameter Count
6. Inference Time
"""

import time

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix
)


NUM_CLASSES = 10


def evaluate_predictions(y_true, y_pred):

    labels = list(range(NUM_CLASSES))

    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "precision": precision,
        "recall": recall,
        "f1_per_class": f1,
        "support": support,
        "confusion_matrix": cm
    }


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def measure_inference_time(
    model,
    sample_input,
    device,
    warmup=20,
    repetitions=100
):
    """
    Measure latency for a fixed input batch.

    Returns average latency per batch in milliseconds.
    """

    model = model.to(device)
    model.eval()

    sample_input = sample_input.to(device)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.inference_mode():

        for _ in range(warmup):
            model(sample_input)

        synchronize()

        start = time.perf_counter()

        for _ in range(repetitions):
            model(sample_input)

        synchronize()

        elapsed = time.perf_counter() - start

    average_ms = (
        elapsed / repetitions * 1000
    )

    return average_ms


def self_test():

    print("=" * 55)
    print("ELEC5305 - Model Metrics Self-Test")
    print("=" * 55)

    # Synthetic labels for metric verification
    y_true = np.array([
        0, 1, 2, 3, 4,
        5, 6, 7, 8, 9
    ])

    y_pred = np.array([
        0, 1, 2, 3, 4,
        5, 6, 7, 8, 0
    ])

    metrics = evaluate_predictions(
        y_true,
        y_pred
    )

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")

    print("\nPer-class Recall:")
    for i, value in enumerate(metrics["recall"]):
        print(f"Class {i}: {value:.4f}")

    assert np.isclose(
        metrics["accuracy"], 0.9
    )

    assert metrics["confusion_matrix"].shape == (
        NUM_CLASSES, NUM_CLASSES
    )

    print("\nClassification metrics: PASSED")

    # Basic inference-time test
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = torch.nn.Linear(80, 10).to(device)

    inputs = torch.randn(
        32, 80,
        device=device
    )

    latency = measure_inference_time(
        model,
        inputs,
        device
    )

    print(f"Inference device: {device}")
    print(f"Average batch latency: {latency:.4f} ms")

    assert latency > 0

    print("\nInference timing test: PASSED")
    print("All model metrics self-tests PASSED!")


if __name__ == "__main__":
    self_test()
