
"""
ELEC5305 Project
UrbanSound8K Data Loading and MFCC Feature Caching

Author: Xiao Hu

Functions:
1. Load UrbanSound8K metadata
2. Extract aggregated MFCC features
3. Preserve the official 10 folds
4. Save features to a compressed NPZ file
5. Run a self-test without the real dataset
"""

from pathlib import Path
import argparse
import tempfile

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from audio_features import (
    SR,
    DURATION,
    N_MFCC,
    N_FFT,
    HOP_LENGTH,
    load_audio,
    extract_mfcc,
    aggregate_mfcc,
    generate_test_audio,
)


# ======================================================
# Section 1: Project Paths
# ======================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "data" / "UrbanSound8K"

METADATA_FILE = (
    DATASET_DIR / "metadata" / "UrbanSound8K.csv"
)

CACHE_DIR = PROJECT_DIR / "data" / "cache"

CACHE_FILE = CACHE_DIR / "mfcc_aggregated_v1.npz"


# ======================================================
# Section 2: Load Metadata
# ======================================================

def load_metadata(metadata_path=METADATA_FILE):

    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {metadata_path}"
        )

    df = pd.read_csv(metadata_path)

    required_columns = [
        "slice_file_name",
        "fold",
        "classID",
        "class"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing metadata column: {column}"
            )

    if df[required_columns].isna().any().any():
        raise ValueError(
            "Missing values in required metadata fields."
        )

    if not df["fold"].between(1, 10).all():
        raise ValueError("Invalid official fold number.")

    if not df["classID"].between(0, 9).all():
        raise ValueError("Invalid class ID.")

    return df


# ======================================================
# Section 3: Extract Features
# ======================================================

def process_dataset(
    dataset_dir=DATASET_DIR,
    metadata_path=METADATA_FILE
):

    dataset_dir = Path(dataset_dir)
    df = load_metadata(metadata_path)

    features = []
    labels = []
    folds = []
    filenames = []

    print("\nExtracting MFCC features...")

    for row in tqdm(
        df.itertuples(index=False),
        total=len(df)
    ):

        fold = int(row.fold)
        label = int(row.classID)
        filename = row.slice_file_name

        audio_path = (
            dataset_dir
            / "audio"
            / f"fold{fold}"
            / filename
        )

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        # Load and preprocess audio
        y, sr = load_audio(audio_path)

        # Extract time-resolved MFCC
        mfcc = extract_mfcc(y)

        # Aggregate mean and standard deviation
        feature_vector = aggregate_mfcc(mfcc)

        if feature_vector.shape != (80,):
            raise ValueError(
                f"Unexpected MFCC feature shape: "
                f"{feature_vector.shape}"
            )

        if not np.isfinite(feature_vector).all():
            raise ValueError(
                f"Non-finite features: {filename}"
            )

        features.append(feature_vector)
        labels.append(label)
        folds.append(fold)
        filenames.append(filename)

    X = np.stack(features).astype(np.float32)

    y = np.asarray(labels, dtype=np.int64)

    fold_ids = np.asarray(folds, dtype=np.int64)

    filenames = np.asarray(filenames)

    return X, y, fold_ids, filenames


# ======================================================
# Section 4: Save Feature Cache
# ======================================================

def save_cache(
    X,
    y,
    fold_ids,
    filenames,
    cache_file=CACHE_FILE
):

    cache_file = Path(cache_file)
    cache_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        cache_file,
        X=X,
        y=y,
        folds=fold_ids,
        filenames=filenames,
        sample_rate=SR,
        duration=DURATION,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    print(f"\nFeature cache saved: {cache_file}")


# ======================================================
# Section 5: Load Feature Cache
# ======================================================

def load_cache(cache_file=CACHE_FILE):

    with np.load(
        cache_file,
        allow_pickle=False
    ) as data:

        X = data["X"]
        y = data["y"]
        folds = data["folds"]

    return X, y, folds


# ======================================================
# Section 6: Validate Dataset
# ======================================================

def validate_dataset(X, y, folds, expected_size=None):

    assert X.ndim == 2
    assert X.shape[1] == 80

    assert len(X) == len(y) == len(folds)

    assert np.isfinite(X).all()

    assert np.all((y >= 0) & (y <= 9))
    assert np.all((folds >= 1) & (folds <= 10))

    if expected_size is not None:

        assert len(X) == expected_size, (
            f"Expected {expected_size} recordings, "
            f"found {len(X)}"
        )

        assert len(np.unique(y)) == 10

        assert np.array_equal(
            np.unique(folds),
            np.arange(1, 11)
        )

    print("\nDataset validation: PASSED")

    print(f"Feature shape: {X.shape}")
    print(f"Label shape: {y.shape}")
    print(f"Fold shape: {folds.shape}")


# ======================================================
# Section 7: Synthetic Data Self-Test
# ======================================================

def self_test():

    print("=" * 55)
    print("ELEC5305 - Dataset Pipeline Self-Test")
    print("=" * 55)

    # Create a temporary mini dataset
    with tempfile.TemporaryDirectory() as temp:

        root = Path(temp) / "UrbanSound8K"

        audio_dir = root / "audio"
        metadata_dir = root / "metadata"

        metadata_dir.mkdir(parents=True)

        y_audio, sr = generate_test_audio()

        records = []

        # Two synthetic audio files in different folds
        for index, fold in enumerate([1, 2]):

            fold_dir = audio_dir / f"fold{fold}"
            fold_dir.mkdir(parents=True)

            filename = f"synthetic_{index}.wav"

            # Write a real temporary WAV file
            sf.write(
                fold_dir / filename,
                y_audio,
                sr,
                subtype="FLOAT"
            )

            records.append({
                "slice_file_name": filename,
                "fold": fold,
                "classID": index,
                "class": f"synthetic_class_{index}"
            })

        metadata_path = (
            metadata_dir / "UrbanSound8K.csv"
        )

        pd.DataFrame(records).to_csv(
            metadata_path,
            index=False
        )

        # Test the real processing functions
        X, y, folds, filenames = process_dataset(
            root,
            metadata_path
        )

        validate_dataset(
            X,
            y,
            folds,
            expected_size=None
        )

        # Verify correct fold preservation
        np.testing.assert_array_equal(
            folds,
            np.array([1, 2])
        )

        np.testing.assert_array_equal(
            y,
            np.array([0, 1])
        )

        # Verify caching and loading
        test_cache = (
            Path(temp) / "test_cache.npz"
        )

        save_cache(
            X,
            y,
            folds,
            filenames,
            test_cache
        )

        X_loaded, y_loaded, folds_loaded = (
            load_cache(test_cache)
        )

        np.testing.assert_allclose(
            X,
            X_loaded
        )

        np.testing.assert_array_equal(
            y,
            y_loaded
        )

        np.testing.assert_array_equal(
            folds,
            folds_loaded
        )

        print("\nCache save/load test: PASSED")
        print("Official fold preservation test: PASSED")
        print("\nAll self-tests PASSED!")


# ======================================================
# Section 8: Main Program
# ======================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--self-test",
        action="store_true"
    )

    parser.add_argument(
        "--build",
        action="store_true"
    )

    args = parser.parse_args()

    if args.self_test:

        self_test()

    elif args.build:

        print("Processing real UrbanSound8K dataset...")

        X, y, folds, filenames = process_dataset()

        validate_dataset(
            X,
            y,
            folds,
            expected_size=8732
        )

        save_cache(
            X,
            y,
            folds,
            filenames
        )

        print("\nReal dataset processing completed!")

    else:

        parser.print_help()


if __name__ == "__main__":
    main()
