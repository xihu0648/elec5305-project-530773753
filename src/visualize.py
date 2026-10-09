
"""
ELEC5305 Project
Environmental Sound Classification

Experiment: Temporal Information Analysis

Research question:
What temporal information is lost when MFCC
features are aggregated using mean and std?

Author: Xiao Hu
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import librosa.display

from audio_features import (
    generate_test_audio,
    extract_mfcc,
    aggregate_mfcc,
    SR,
    HOP_LENGTH
)


# ============================================================
# Section 1: Project Configuration
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_DIR / "results" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42


# ============================================================
# Section 2: Shuffle MFCC Temporal Frames
# ============================================================

def shuffle_mfcc_frames(mfcc):
    """
    Randomly reorder the time frames of MFCCs.

    The same permutation is applied to all coefficients.

    This preserves the values within each original frame
    but destroys their temporal ordering.
    """

    rng = np.random.default_rng(RANDOM_SEED)

    num_frames = mfcc.shape[1]

    permutation = rng.permutation(num_frames)

    shuffled_mfcc = mfcc[:, permutation]

    return shuffled_mfcc, permutation


# ============================================================
# Section 3: Compare Aggregated Features
# ============================================================

def compare_aggregated_features(original, shuffled):

    original_features = aggregate_mfcc(original)
    shuffled_features = aggregate_mfcc(shuffled)

    difference = np.abs(
        original_features - shuffled_features
    )

    max_difference = np.max(difference)
    mean_difference = np.mean(difference)

    print("\nAggregated Feature Comparison")
    print("-" * 45)

    print(
        f"Original feature shape: "
        f"{original_features.shape}"
    )

    print(
        f"Shuffled feature shape: "
        f"{shuffled_features.shape}"
    )

    print(
        f"Maximum absolute difference: "
        f"{max_difference:.10f}"
    )

    print(
        f"Mean absolute difference: "
        f"{mean_difference:.10f}"
    )

    # The aggregated features should be numerically equal.
    np.testing.assert_allclose(
        original_features,
        shuffled_features,
        rtol=1e-5,
        atol=1e-4
    )

    print("\nAggregation invariance test: PASSED")

    return original_features, shuffled_features, difference


# ============================================================
# Section 4: Temporal Structure Visualisation
# ============================================================

def plot_temporal_comparison(
    original_mfcc,
    shuffled_mfcc,
    original_features,
    shuffled_features,
    difference
):

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(13, 15)
    )

    # Use the same colour scale for a fair comparison.
    vmin = min(
        original_mfcc.min(),
        shuffled_mfcc.min()
    )

    vmax = max(
        original_mfcc.max(),
        shuffled_mfcc.max()
    )

    # --------------------------------------------------------
    # Figure 1: Original MFCC
    # --------------------------------------------------------

    img1 = librosa.display.specshow(
        original_mfcc,
        sr=SR,
        hop_length=HOP_LENGTH,
        x_axis="time",
        ax=axes[0],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax
    )

    axes[0].set_title(
        "Original MFCC - Temporal Structure Preserved"
    )

    axes[0].set_ylabel("MFCC Coefficient")

    fig.colorbar(img1, ax=axes[0])

    # --------------------------------------------------------
    # Figure 2: Shuffled MFCC
    # --------------------------------------------------------

    img2 = librosa.display.specshow(
        shuffled_mfcc,
        sr=SR,
        hop_length=HOP_LENGTH,
        x_axis="time",
        ax=axes[1],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax
    )

    axes[1].set_title(
        "Shuffled MFCC - Temporal Order Destroyed"
    )

    axes[1].set_ylabel("MFCC Coefficient")

    fig.colorbar(img2, ax=axes[1])

    # --------------------------------------------------------
    # Figure 3: Aggregated MFCC Comparison
    # --------------------------------------------------------

    feature_indices = np.arange(
        len(original_features)
    )

    axes[2].plot(
        feature_indices,
        original_features,
        label="Original MFCC Statistics",
        linewidth=2
    )

    axes[2].plot(
        feature_indices,
        shuffled_features,
        label="Shuffled MFCC Statistics",
        linestyle="--",
        linewidth=1.5
    )

    axes[2].set_title(
        "Aggregated MFCC Features - Mean and Std"
    )

    axes[2].set_xlabel("Feature Index")
    axes[2].set_ylabel("Feature Value")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    # --------------------------------------------------------
    # Figure 4: Absolute Difference
    # --------------------------------------------------------

    axes[3].plot(
        feature_indices,
        difference,
        linewidth=1.5
    )

    axes[3].set_title(
        "Absolute Difference Between Aggregated Features"
    )

    axes[3].set_xlabel("Feature Index")
    axes[3].set_ylabel("Absolute Difference")
    axes[3].grid(alpha=0.3)

    plt.tight_layout()

    output_file = (
        RESULTS_DIR / "temporal_information_comparison.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    print(f"\nFigure saved to: {output_file}")


# ============================================================
# Section 5: Main Experiment
# ============================================================

def main():

    print("=" * 55)
    print("ELEC5305 - Temporal Information Analysis")
    print("=" * 55)

    # Step 1: Generate test signal
    y, sr = generate_test_audio()

    print("\nSynthetic audio generated.")
    print(f"Sampling rate: {sr} Hz")
    print(f"Duration: {len(y) / sr:.2f} seconds")

    # Step 2: Extract original MFCC
    original_mfcc = extract_mfcc(y)

    print(
        f"\nOriginal MFCC shape: "
        f"{original_mfcc.shape}"
    )

    # Step 3: Shuffle time frames
    shuffled_mfcc, permutation = shuffle_mfcc_frames(
        original_mfcc
    )

    print(
        f"Shuffled MFCC shape: "
        f"{shuffled_mfcc.shape}"
    )

    print(
        f"Number of shuffled frames: "
        f"{len(permutation)}"
    )

    # Verify that frame order has changed
    assert not np.array_equal(
        permutation,
        np.arange(len(permutation))
    )

    # Step 4: Compare aggregated features
    original_features, shuffled_features, difference = (
        compare_aggregated_features(
            original_mfcc,
            shuffled_mfcc
        )
    )

    # Step 5: Create figures
    plot_temporal_comparison(
        original_mfcc,
        shuffled_mfcc,
        original_features,
        shuffled_features,
        difference
    )

    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()
