
"""
ELEC5305 Project
Environmental Sound Classification

Module: Audio Feature Extraction

Functions:
1. Audio preprocessing
2. STFT computation
3. MFCC extraction
4. Log-Mel spectrogram extraction
5. Feature aggregation
6. Time-frequency visualisation

Author: Xiao Hu
"""

from pathlib import Path

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


# ============================================================
# Section 1: Project Configuration
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_DIR / "results" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SR = 22050
DURATION = 4.0

N_FFT = 1024
HOP_LENGTH = 512

N_MFCC = 40
N_MELS = 128


# ============================================================
# Section 2: Audio Preprocessing
# ============================================================

def load_audio(file_path):
    """
    Load audio as mono at 22050 Hz.

    Preserve the original amplitude scale instead of
    applying peak normalisation to each recording.
    """

    y, sr = librosa.load(
        file_path,
        sr=SR,
        mono=True
    )

    target_length = int(SR * DURATION)

    if len(y) < target_length:
        y = np.pad(
            y,
            (0, target_length - len(y))
        )

    else:
        y = y[:target_length]

    return y, sr


# ============================================================
# Section 3: Generate Synthetic Test Audio
# ============================================================

def generate_test_audio():
    """
    Generate a synthetic signal containing:
    1. A low-frequency tone
    2. A chirp
    3. Repeated short bursts

    Used to verify the feature extraction pipeline.
    """

    t = np.arange(int(SR * DURATION)) / SR

    # Component 1: 440 Hz sinusoidal signal
    tone = 0.20 * np.sin(2 * np.pi * 440 * t)

    # Component 2: Frequency-modulated chirp
    chirp_phase = 2 * np.pi * (
        300 * t + 0.5 * 400 * t**2
    )

    chirp = 0.15 * np.sin(chirp_phase)

    # Component 3: Repeated bursts
    bursts = np.zeros_like(t)

    for start in np.arange(0.4, DURATION, 0.6):
        mask = (t >= start) & (t < start + 0.12)

        envelope = np.hanning(np.count_nonzero(mask))

        bursts[mask] += (
            0.25
            * envelope
            * np.sin(2 * np.pi * 1800 * t[mask])
        )

    # Combine all components
    y = tone + chirp + bursts

    return y.astype(np.float32), SR


# ============================================================
# Section 4: STFT
# ============================================================

def extract_stft(y):

    stft_complex = librosa.stft(
        y,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window="hann"
    )

    magnitude = np.abs(stft_complex)

    stft_db = librosa.amplitude_to_db(
        magnitude,
        ref=1.0
    )

    return stft_complex, stft_db


# ============================================================
# Section 5: MFCC Feature Extraction
# ============================================================

def extract_mfcc(y):

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=SR,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    return mfcc


# ============================================================
# Section 6: Log-Mel Spectrogram
# ============================================================

def extract_logmel(y):

    mel_power = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )

    logmel = librosa.power_to_db(
        mel_power,
        ref=1.0
    )

    return logmel


# ============================================================
# Section 7: Aggregated MFCC Features
# ============================================================

def aggregate_mfcc(mfcc):
    """
    Calculate mean and standard deviation over time.

    Input:
        (40, number_of_frames)

    Output:
        (80,)
    """

    mfcc_mean = np.mean(mfcc, axis=1)

    mfcc_std = np.std(mfcc, axis=1)

    features = np.concatenate(
        [mfcc_mean, mfcc_std]
    )

    return features.astype(np.float32)


# ============================================================
# Section 8: Visualisation
# ============================================================

def plot_audio_features(y, sr, stft_db, mfcc, logmel):

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, 14)
    )

    # Figure A: Waveform
    librosa.display.waveshow(
        y,
        sr=sr,
        ax=axes[0]
    )

    axes[0].set_title(
        "Synthetic Audio - Time Domain Waveform"
    )

    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")

    # Figure B: STFT
    img1 = librosa.display.specshow(
        stft_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="linear",
        ax=axes[1],
        cmap="magma"
    )

    axes[1].set_title("STFT Spectrogram")

    fig.colorbar(
        img1,
        ax=axes[1],
        format="%+2.0f dB"
    )

    # Figure C: MFCC
    img2 = librosa.display.specshow(
        mfcc,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        ax=axes[2],
        cmap="viridis"
    )

    axes[2].set_title("MFCC - 40 Coefficients")
    axes[2].set_ylabel("MFCC Index")

    fig.colorbar(
        img2,
        ax=axes[2]
    )

    # Figure D: Log-Mel
    img3 = librosa.display.specshow(
        logmel,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        ax=axes[3],
        cmap="magma"
    )

    axes[3].set_title(
        "Log-Mel Spectrogram - 128 Mel Bands"
    )

    fig.colorbar(
        img3,
        ax=axes[3],
        format="%+2.0f dB"
    )

    plt.tight_layout()

    output_file = RESULTS_DIR / "synthetic_audio_features.png"

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    print(f"Figure saved to: {output_file}")


# ============================================================
# Section 9: Main Program
# ============================================================

def main():

    print("=" * 55)
    print("ELEC5305 - Audio Feature Extraction")
    print("=" * 55)

    print("\nGenerating synthetic test audio...")

    y, sr = generate_test_audio()

    print(f"Sampling rate: {sr} Hz")
    print(f"Audio duration: {len(y) / sr:.2f} seconds")
    print(f"Number of samples: {len(y)}")

    # Extract features
    _, stft_db = extract_stft(y)

    mfcc = extract_mfcc(y)

    logmel = extract_logmel(y)

    aggregated = aggregate_mfcc(mfcc)

    # Print dimensions
    print("\nFeature Dimensions:")
    print(f"STFT: {stft_db.shape}")
    print(f"MFCC: {mfcc.shape}")
    print(f"Log-Mel: {logmel.shape}")
    print(f"Aggregated MFCC: {aggregated.shape}")

    # Visualisation
    plot_audio_features(
        y,
        sr,
        stft_db,
        mfcc,
        logmel
    )

    print("\nFeature extraction completed successfully!")


if __name__ == "__main__":
    main()
