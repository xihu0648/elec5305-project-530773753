
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ==================================================
# ELEC5305 Project
# UrbanSound8K Dataset Inspection
# ==================================================

# Project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "data" / "UrbanSound8K"
METADATA_FILE = DATASET_DIR / "metadata" / "UrbanSound8K.csv"
RESULTS_DIR = PROJECT_DIR / "results" / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 50)
    print("UrbanSound8K Dataset Inspection")
    print("=" * 50)

    # Step 1: Check metadata
    if not METADATA_FILE.exists():
        print("Dataset not found!")
        print(f"Expected file: {METADATA_FILE}")
        print("Please download and extract UrbanSound8K.")
        return

    # Step 2: Load dataset metadata
    df = pd.read_csv(METADATA_FILE)

    print("\nDataset Information:")
    print(f"Total recordings: {len(df)}")
    print(f"Number of classes: {df['class'].nunique()}")
    print(f"Number of folds: {df['fold'].nunique()}")

    # Step 3: Class distribution
    class_counts = df["class"].value_counts().sort_index()

    print("\nClass Distribution:")
    print(class_counts)

    plt.figure(figsize=(12, 6))
    class_counts.plot(kind="bar")

    plt.title("UrbanSound8K Class Distribution")
    plt.xlabel("Sound Class")
    plt.ylabel("Number of Recordings")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "class_distribution.png",
        dpi=300
    )
    plt.close()

    # Step 4: Official fold distribution
    fold_counts = df["fold"].value_counts().sort_index()

    print("\nOfficial Fold Distribution:")
    print(fold_counts)

    plt.figure(figsize=(10, 5))
    fold_counts.plot(kind="bar")

    plt.title("UrbanSound8K Official Fold Distribution")
    plt.xlabel("Fold Number")
    plt.ylabel("Number of Recordings")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "fold_distribution.png",
        dpi=300
    )
    plt.close()

    # Step 5: Basic validation
    assert len(df) == 8732, "Unexpected number of recordings!"
    assert df["class"].nunique() == 10, "Expected 10 classes!"
    assert sorted(df["fold"].unique()) == list(range(1, 11)), \
        "Expected official folds 1 through 10!"

    print("\nDataset validation passed!")
    print(f"Figures saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
