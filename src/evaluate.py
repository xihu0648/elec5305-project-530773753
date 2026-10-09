
"""
ELEC5305 Project
Official UrbanSound8K 10-Fold Evaluation

Models:
A - Aggregated MFCC + MLP
B - Time-resolved MFCC + CNN
C - Log-Mel Spectrogram + CNN

Author: Xiao Hu

Usage:
    python src/evaluate.py --self-test
    python src/evaluate.py --dry-run
    python src/evaluate.py --run
    python src/evaluate.py --summarize
"""

from pathlib import Path
import argparse
import csv
import json
import subprocess
import sys
import tempfile

import numpy as np
import matplotlib.pyplot as plt


# =====================================================
# Section 1: Project Configuration
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
SUMMARY_DIR = RESULTS_DIR / "cross_validation"

MODELS = ["mlp", "mfcc_cnn", "logmel_cnn"]
FOLDS = list(range(1, 11))


# =====================================================
# Section 2: Official Fold Schedule
# =====================================================

def get_fold_split(test_fold):

    if test_fold not in FOLDS:
        raise ValueError("Invalid test fold")

    val_fold = (test_fold % 10) + 1

    train_folds = [
        f for f in FOLDS
        if f not in (test_fold, val_fold)
    ]

    return train_folds, val_fold, test_fold


def validate_fold_schedule():

    for test_fold in FOLDS:

        train, val, test = get_fold_split(test_fold)

        assert len(train) == 8
        assert val != test
        assert test not in train
        assert val not in train

        assert set(train + [val, test]) == set(FOLDS)

    print("Official fold schedule: PASSED")


# =====================================================
# Section 3: Experiment Commands
# =====================================================

def build_command(model, fold):

    python = sys.executable

    if model == "mlp":

        return [
            python,
            str(SRC_DIR / "train_mlp.py"),
            "--fold",
            str(fold)
        ]

    if model in ("mfcc_cnn", "logmel_cnn"):

        representation = (
            "mfcc" if model == "mfcc_cnn"
            else "logmel"
        )

        return [
            python,
            str(SRC_DIR / "train_cnn.py"),
            "--representation",
            representation,
            "--fold",
            str(fold)
        ]

    raise ValueError(f"Unknown model: {model}")


# =====================================================
# Section 4: Result File Locations
# =====================================================

def metric_path(model, fold):

    if model == "mlp":

        return (
            RESULTS_DIR
            / "mlp"
            / f"fold{fold}"
            / "metrics.json"
        )

    representation = (
        "mfcc" if model == "mfcc_cnn"
        else "logmel"
    )

    return (
        RESULTS_DIR
        / "cnn"
        / "real"
        / representation
        / f"fold{fold}"
        / "metrics.json"
    )


# =====================================================
# Section 5: Validate Real Feature Caches
# =====================================================

def validate_real_caches():

    cache_files = [
        DATA_DIR / "cache" / "mfcc_aggregated_v1.npz",
        DATA_DIR / "cache" / "mfcc_temporal_v1.npz",
        DATA_DIR / "cache" / "logmel_temporal_v1.npz"
    ]

    reference_labels = None
    reference_folds = None

    for cache_file in cache_files:

        if not cache_file.exists():
            raise FileNotFoundError(
                f"Missing feature cache: {cache_file}"
            )

        with np.load(
            cache_file,
            allow_pickle=False
        ) as data:

            y = data["y"]
            folds = data["folds"]
            X_shape = data["X"].shape

        assert len(y) == 8732
        assert len(folds) == 8732
        assert X_shape[0] == 8732

        np.testing.assert_array_equal(
            np.unique(folds),
            FOLDS
        )

        if reference_labels is None:
            reference_labels = y.copy()
            reference_folds = folds.copy()
        else:
            np.testing.assert_array_equal(
                y, reference_labels
            )
            np.testing.assert_array_equal(
                folds, reference_folds
            )

        print(f"Validated: {cache_file.name}")

    print("All three feature caches: PASSED")


# =====================================================
# Section 6: Run All Experiments
# =====================================================

def run_all(dry_run=False):

    validate_fold_schedule()

    if not dry_run:
        validate_real_caches()

    total = len(MODELS) * len(FOLDS)
    run_number = 0

    for model in MODELS:

        for fold in FOLDS:

            run_number += 1

            command = build_command(model, fold)

            print("\n" + "=" * 60)
            print(
                f"Experiment {run_number}/{total}: "
                f"{model}, Test Fold {fold}"
            )
            print("=" * 60)

            print("Command:", " ".join(command), flush=True)

            if dry_run:
                continue

            result_file = metric_path(model, fold)

            # Skip already completed real runs.
            if result_file.exists():

                with open(
                    result_file,
                    encoding="utf-8"
                ) as f:
                    existing = json.load(f)

                if (
                    existing.get("synthetic_test") is False
                    and existing.get("test_fold") == fold
                ):
                    print("Existing result found. Skipping.")
                    continue

            subprocess.run(
                command,
                cwd=PROJECT_DIR,
                check=True
            )

            if not result_file.exists():
                raise RuntimeError(
                    f"Result not saved: {result_file}"
                )

    if dry_run:
        print("\nDry-run completed. No models trained.")
    else:
        print("\nAll requested experiments completed.")


# =====================================================
# Section 7: Load and Summarize Results
# =====================================================

def summarize_results(
    output_dir=SUMMARY_DIR,
    metrics_root=RESULTS_DIR
):

    output_dir = Path(output_dir)
    metrics_root = Path(metrics_root)

    rows = []
    summary = []

    for model in MODELS:

        accuracies = []
        macro_f1_scores = []

        for fold in FOLDS:

            if model == "mlp":
                filename = (
                    metrics_root / "mlp"
                    / f"fold{fold}" / "metrics.json"
                )

            else:
                rep = (
                    "mfcc" if model == "mfcc_cnn"
                    else "logmel"
                )

                filename = (
                    metrics_root / "cnn" / "real"
                    / rep / f"fold{fold}" / "metrics.json"
                )

            if not filename.exists():
                raise FileNotFoundError(
                    f"Missing fold result: {filename}"
                )

            with open(
                filename,
                encoding="utf-8"
            ) as f:
                metrics = json.load(f)

            if metrics.get("synthetic_test") is not False:
                raise ValueError(
                    f"Not a real result: {filename}"
                )

            if metrics["test_fold"] != fold:
                raise ValueError(
                    f"Fold mismatch: {filename}"
                )

            accuracy = float(metrics["accuracy"])
            macro_f1 = float(metrics["macro_f1"])

            if not (
                0 <= accuracy <= 1
                and 0 <= macro_f1 <= 1
            ):
                raise ValueError("Invalid metric range")

            accuracies.append(accuracy)
            macro_f1_scores.append(macro_f1)

            rows.append({
                "model": model,
                "test_fold": fold,
                "accuracy": accuracy,
                "macro_f1": macro_f1
            })

        summary.append({
            "model": model,
            "accuracy_mean": float(
                np.mean(accuracies)
            ),
            "accuracy_std": float(
                np.std(accuracies, ddof=1)
            ),
            "macro_f1_mean": float(
                np.mean(macro_f1_scores)
            ),
            "macro_f1_std": float(
                np.std(macro_f1_scores, ddof=1)
            )
        })

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save fold-by-fold CSV
    with open(
        output_dir / "fold_results.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model", "test_fold",
                "accuracy", "macro_f1"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    # Save summary JSON
    with open(
        output_dir / "summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=4)

    # Plot accuracy and macro-F1
    names = [item["model"] for item in summary]
    x = np.arange(len(names))

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5)
    )

    for ax, mean_key, std_key, title in [
        (
            axes[0], "accuracy_mean",
            "accuracy_std", "Accuracy"
        ),
        (
            axes[1], "macro_f1_mean",
            "macro_f1_std", "Macro-F1"
        )
    ]:

        means = [
            item[mean_key] * 100
            for item in summary
        ]

        stds = [
            item[std_key] * 100
            for item in summary
        ]

        ax.bar(
            x,
            means,
            yerr=stds,
            capsize=5
        )

        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Score (%)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        output_dir / "model_comparison.png",
        dpi=300
    )

    plt.close(fig)

    print("\nOFFICIAL 10-FOLD SUMMARY")
    print("=" * 60)

    for item in summary:

        print(
            f"{item['model']}: "
            f"Accuracy = "
            f"{item['accuracy_mean']:.4f} "
            f"+/- {item['accuracy_std']:.4f}, "
            f"Macro-F1 = "
            f"{item['macro_f1_mean']:.4f} "
            f"+/- {item['macro_f1_std']:.4f}"
        )

    print(f"\nResults saved to: {output_dir}")


# =====================================================
# Section 8: Self-Test
# =====================================================

def self_test():

    print("ELEC5305 - Evaluation Self-Test")
    print("=" * 55)

    validate_fold_schedule()

    # Verify 30 experiment commands.
    commands = [
        build_command(model, fold)
        for model in MODELS
        for fold in FOLDS
    ]

    assert len(commands) == 30

    print("Experiment command count: 30")
    print("Command generation: PASSED")

    # Create temporary example metrics solely to test
    # summary file handling and statistics.
    with tempfile.TemporaryDirectory() as tmp:

        root = Path(tmp)
        metrics_root = root / "results"

        for model in MODELS:

            for fold in FOLDS:

                if model == "mlp":
                    folder = (
                        metrics_root / "mlp"
                        / f"fold{fold}"
                    )
                else:
                    rep = (
                        "mfcc" if model == "mfcc_cnn"
                        else "logmel"
                    )

                    folder = (
                        metrics_root / "cnn"
                        / "real" / rep
                        / f"fold{fold}"
                    )

                folder.mkdir(parents=True)

                # These are test fixtures, not measured
                # model results.
                fixture = {
                    "test_fold": fold,
                    "accuracy": 0.5,
                    "macro_f1": 0.4,
                    "synthetic_test": False
                }

                with open(
                    folder / "metrics.json",
                    "w",
                    encoding="utf-8"
                ) as f:
                    json.dump(fixture, f)

        summarize_results(
            output_dir=root / "self_test_output",
            metrics_root=metrics_root
        )

        assert (
            root / "self_test_output"
            / "fold_results.csv"
        ).exists()

        assert (
            root / "self_test_output"
            / "model_comparison.png"
        ).exists()

    print("\nAll evaluation self-tests PASSED!")
    print("No real model training was performed.")
    print("Temporary example metrics were discarded.")


# =====================================================
# Section 9: Main
# =====================================================

def main():

    parser = argparse.ArgumentParser()

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--self-test",
        action="store_true"
    )

    mode.add_argument(
        "--dry-run",
        action="store_true"
    )

    mode.add_argument(
        "--run",
        action="store_true"
    )

    mode.add_argument(
        "--summarize",
        action="store_true"
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()

    elif args.dry_run:
        run_all(dry_run=True)

    elif args.run:
        run_all(dry_run=False)
        summarize_results()

    elif args.summarize:
        summarize_results()


if __name__ == "__main__":
    main()
