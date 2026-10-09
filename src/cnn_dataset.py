
"""
ELEC5305 Project
CNN Feature Dataset Preparation

Model B: Time-resolved MFCC + CNN
Model C: Log-Mel Spectrogram + CNN

Features:
- Audio loading and preprocessing
- MFCC and Log-Mel extraction
- Official fold preservation
- Feature caching
- Synthetic data self-test

Author: Xiao Hu
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
    N_FFT,
    HOP_LENGTH,
    N_MFCC,
    N_MELS,
    load_audio,
    extract_mfcc,
    extract_logmel,
    generate_test_audio
)


# ==================================================
# Section 1: Paths and Configuration
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_DIR / "data" / "UrbanSound8K"
)

METADATA_FILE = (
    DATASET_DIR / "metadata" / "UrbanSound8K.csv"
)

CACHE_DIR = PROJECT_DIR / "data" / "cache"

MFCC_CACHE = CACHE_DIR / "mfcc_temporal_v1.npz"

LOGMEL_CACHE = CACHE_DIR / "logmel_temporal_v1.npz"

TARGET_LENGTH = int(SR * DURATION)

# librosa STFT uses center=True by default.
EXPECTED_FRAMES = (
    1 + TARGET_LENGTH // HOP_LENGTH
)


# ==================================================
# Section 2: Load Metadata
# ==================================================

def load_metadata(metadata_file):

    metadata_file = Path(metadata_file)

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata not found: {metadata_file}"
        )

    df = pd.read_csv(metadata_file)

    required = [
        "slice_file_name",
        "fold",
        "classID",
        "class"
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Missing metadata column: {col}"
            )

    if df[required].isna().any().any():
        raise ValueError(
            "Metadata contains missing values."
        )

    if not df["fold"].between(1, 10).all():
        raise ValueError("Invalid fold number.")

    if not df["classID"].between(0, 9).all():
        raise ValueError("Invalid class ID.")

    return df


# ==================================================
# Section 3: Extract CNN Features
# ==================================================

def extract_cnn_feature(audio, representation):

    if representation == "mfcc":

        feature = extract_mfcc(audio)

        expected_shape = (
            N_MFCC,
            EXPECTED_FRAMES
        )

    elif representation == "logmel":

        feature = extract_logmel(audio)

        expected_shape = (
            N_MELS,
            EXPECTED_FRAMES
        )

    else:
        raise ValueError(
            "Representation must be mfcc or logmel."
        )

    feature = feature.astype(np.float32)

    if feature.shape != expected_shape:
        raise ValueError(
            f"Unexpected shape: {feature.shape}, "
            f"expected: {expected_shape}"
        )

    if not np.isfinite(feature).all():
        raise ValueError(
            "Feature contains NaN or infinity."
        )

    return feature


# ==================================================
# Section 4: Process Dataset
# ==================================================

def process_dataset(
    dataset_dir,
    metadata_file,
    representation
):

    df = load_metadata(metadata_file)

    dataset_dir = Path(dataset_dir)

    features = []
    labels = []
    folds = []
    filenames = []

    print(
        f"\nExtracting {representation} features..."
    )

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
                f"Missing audio: {audio_path}"
            )

        audio, _ = load_audio(audio_path)

        feature = extract_cnn_feature(
            audio,
            representation
        )

        features.append(feature)
        labels.append(label)
        folds.append(fold)
        filenames.append(filename)

    X = np.stack(features).astype(np.float32)

    y = np.asarray(labels, dtype=np.int64)

    fold_ids = np.asarray(
        folds, dtype=np.int64
    )

    filenames = np.asarray(filenames)

    return X, y, fold_ids, filenames


# ==================================================
# Section 5: Save and Load Feature Cache
# ==================================================

def save_cache(
    X,
    y,
    folds,
    filenames,
    cache_file,
    representation
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
        folds=folds,
        filenames=filenames,
        representation=representation,
        sample_rate=SR,
        duration=DURATION,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    print(f"\nCache saved: {cache_file}")


def load_cache(cache_file):

    with np.load(
        cache_file,
        allow_pickle=False
    ) as data:

        X = data["X"]
        y = data["y"]
        folds = data["folds"]

    return X, y, folds


# ==================================================
# Section 6: Validate Features
# ==================================================

def validate_features(
    X,
    y,
    folds,
    representation,
    expected_size=None
):

    if representation == "mfcc":
        expected_rows = N_MFCC

    elif representation == "logmel":
        expected_rows = N_MELS

    else:
        raise ValueError("Invalid representation.")

    assert X.ndim == 3

    assert X.shape[1:] == (
        expected_rows,
        EXPECTED_FRAMES
    )

    assert len(X) == len(y) == len(folds)

    assert np.isfinite(X).all()

    assert np.all((y >= 0) & (y <= 9))

    assert np.all((folds >= 1) & (folds <= 10))

    if expected_size is not None:

        assert len(X) == expected_size

        assert len(np.unique(y)) == 10

        np.testing.assert_array_equal(
            np.unique(folds),
            np.arange(1, 11)
        )

    print("\nFeature validation: PASSED")

    print(f"Feature shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Folds shape: {folds.shape}")


# ==================================================
# Section 7: Synthetic Self-Test
# ==================================================

def self_test():

    print("=" * 55)
    print("ELEC5305 - CNN Dataset Self-Test")
    print("=" * 55)

    with tempfile.TemporaryDirectory() as temp:

        root = Path(temp) / "UrbanSound8K"

        metadata_dir = root / "metadata"
        metadata_dir.mkdir(parents=True)

        audio, sr = generate_test_audio()

        records = []

        for index, fold in enumerate([1, 2]):

            fold_dir = (
                root / "audio" / f"fold{fold}"
            )

            fold_dir.mkdir(parents=True)

            filename = f"test_{index}.wav"

            sf.write(
                fold_dir / filename,
                audio,
                sr,
                subtype="FLOAT"
            )

            records.append({
                "slice_file_name": filename,
                "fold": fold,
                "classID": index,
                "class": f"test_class_{index}"
            })

        metadata_file = (
            metadata_dir / "UrbanSound8K.csv"
        )

        pd.DataFrame(records).to_csv(
            metadata_file,
            index=False
        )

        for representation in ["mfcc", "logmel"]:

            print(
                f"\nTesting representation: "
                f"{representation}"
            )

            X, y, folds, filenames = (
                process_dataset(
                    root,
                    metadata_file,
                    representation
                )
            )

            validate_features(
                X,
                y,
                folds,
                representation
            )

            test_cache = (
                Path(temp)
                / f"{representation}_test.npz"
            )

            save_cache(
                X,
                y,
                folds,
                filenames,
                test_cache,
                representation
            )

            X2, y2, folds2 = load_cache(
                test_cache
            )

            np.testing.assert_allclose(X, X2)

            np.testing.assert_array_equal(
                y, y2
            )

            np.testing.assert_array_equal(
                folds, folds2
            )

            np.testing.assert_array_equal(
                folds,
                np.array([1, 2])
            )

            print(
                f"{representation} cache test: PASSED"
            )

    print("\nAll CNN dataset self-tests PASSED!")


# ==================================================
# Section 8: Main
# ==================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--self-test",
        action="store_true"
    )

    parser.add_argument(
        "--build",
        choices=["mfcc", "logmel", "both"]
    )

    args = parser.parse_args()

    if args.self_test:

        self_test()

    elif args.build:

        if args.build == "both":
            representations = ["mfcc", "logmel"]

        else:
            representations = [args.build]

        for representation in representations:

            X, y, folds, filenames = (
                process_dataset(
                    DATASET_DIR,
                    METADATA_FILE,
                    representation
                )
            )

            validate_features(
                X,
                y,
                folds,
                representation,
                expected_size=8732
            )

            cache_file = (
                MFCC_CACHE
                if representation == "mfcc"
                else LOGMEL_CACHE
            )

            save_cache(
                X,
                y,
                folds,
                filenames,
                cache_file,
                representation
            )

            print(
                f"\n{representation} dataset completed!"
            )

    else:

        parser.print_help()


if __name__ == "__main__":
    main()
