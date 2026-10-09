
"""
ELEC5305 Project
Real UrbanSound8K Audio Case Analysis

Analyses:
1. Waveform
2. STFT Spectrogram
3. MFCC Trajectories
4. Log-Mel Spectrogram
5. RMS Envelope
6. Temporal Modulation Spectrum

All audio examples come from the real UrbanSound8K dataset.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt


# ==================================================
# 1. Configuration
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT / "data" / "UrbanSound8K"
METADATA_FILE = DATASET_DIR / "metadata" / "UrbanSound8K.csv"

OUTPUT_DIR = ROOT / "results" / "audio_cases"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SR = 22050
DURATION = 4.0
N_FFT = 1024
HOP_LENGTH = 512
N_MFCC = 40
N_MELS = 128

CLASSES = [
    "air_conditioner",
    "engine_idling",
    "drilling",
    "jackhammer"
]


# ==================================================
# 2. Select Real Audio Recordings
# ==================================================

def select_recordings():

    df = pd.read_csv(METADATA_FILE)

    selected = []

    for class_name in CLASSES:

        subset = df[
            df["class"] == class_name
        ].sort_values(
            ["fold", "slice_file_name"]
        )

        if subset.empty:
            raise ValueError(
                f"No recordings for {class_name}"
            )

        # Deterministic selection.
        row = subset.iloc[0]

        audio_path = (
            DATASET_DIR
            / "audio"
            / f"fold{int(row['fold'])}"
            / row["slice_file_name"]
        )

        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        selected.append({
            "class": class_name,
            "fold": int(row["fold"]),
            "filename": row["slice_file_name"],
            "path": audio_path
        })

    return selected


# ==================================================
# 3. Audio Loading
# ==================================================

def load_recording(path):

    audio, _ = librosa.load(
        path,
        sr=SR,
        mono=True
    )

    target_length = int(SR * DURATION)

    audio = librosa.util.fix_length(
        audio,
        size=target_length
    )

    return audio


# ==================================================
# 4. Feature Extraction
# ==================================================

def extract_features(audio):

    stft = librosa.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window="hann"
    )

    stft_db = librosa.amplitude_to_db(
        np.abs(stft),
        ref=np.max
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    logmel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    mfcc = librosa.feature.mfcc(
        S=logmel,
        n_mfcc=N_MFCC
    )

    rms = librosa.feature.rms(
        y=audio,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH
    )[0]

    frame_times = librosa.frames_to_time(
        np.arange(len(rms)),
        sr=SR,
        hop_length=HOP_LENGTH
    )

    # Remove mean from the envelope.
    rms_centered = rms - np.mean(rms)

    # Envelope sample rate is SR / HOP_LENGTH.
    envelope_rate = SR / HOP_LENGTH

    modulation = np.abs(
        np.fft.rfft(rms_centered)
    )

    modulation_freq = np.fft.rfftfreq(
        len(rms_centered),
        d=1.0 / envelope_rate
    )

    return {
        "stft_db": stft_db,
        "logmel": logmel,
        "mfcc": mfcc,
        "rms": rms,
        "frame_times": frame_times,
        "modulation": modulation,
        "modulation_freq": modulation_freq
    }


# ==================================================
# 5. Plot Audio Analysis
# ==================================================

def plot_audio_case(record, audio, features):

    class_name = record["class"]

    fig, axes = plt.subplots(
        6, 1,
        figsize=(14, 19),
        constrained_layout=True
    )

    times = np.arange(len(audio)) / SR

    # Waveform
    axes[0].plot(
        times,
        audio,
        linewidth=0.5
    )

    axes[0].set_title(
        f"{class_name} - Waveform"
    )
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(0, DURATION)

    # STFT
    librosa.display.specshow(
        features["stft_db"],
        sr=SR,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="linear",
        ax=axes[1],
        cmap="magma"
    )

    axes[1].set_title("STFT Spectrogram (dB)")

    # MFCC
    librosa.display.specshow(
        features["mfcc"],
        sr=SR,
        hop_length=HOP_LENGTH,
        x_axis="time",
        ax=axes[2],
        cmap="viridis"
    )

    axes[2].set_title("MFCC - 40 Coefficients")
    axes[2].set_ylabel("MFCC Index")

    # Log-Mel
    librosa.display.specshow(
        features["logmel"],
        sr=SR,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        ax=axes[3],
        cmap="magma"
    )

    axes[3].set_title(
        "Log-Mel Spectrogram - 128 Bands"
    )

    # RMS Envelope
    axes[4].plot(
        features["frame_times"],
        features["rms"]
    )

    axes[4].set_title("RMS Energy Envelope")
    axes[4].set_xlabel("Time (s)")
    axes[4].set_ylabel("RMS")
    axes[4].set_xlim(0, DURATION)

    # Temporal modulation spectrum
    frequencies = features["modulation_freq"]
    amplitudes = features["modulation"]

    mask = (
        (frequencies > 0)
        & (frequencies <= 20)
    )

    axes[5].plot(
        frequencies[mask],
        amplitudes[mask]
    )

    axes[5].set_title(
        "RMS Envelope Modulation Spectrum"
    )
    axes[5].set_xlabel(
        "Modulation Frequency (Hz)"
    )
    axes[5].set_ylabel("Magnitude")
    axes[5].set_xlim(0, 20)

    fig.suptitle(
        f"UrbanSound8K: {class_name}\n"
        f"Fold {record['fold']} | "
        f"{record['filename']}",
        fontsize=15
    )

    output_file = (
        OUTPUT_DIR / f"{class_name}_analysis.png"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Figure saved: {output_file}")


# ==================================================
# 6. Main
# ==================================================

def main():

    print("=" * 60)
    print("ELEC5305 - Real Audio Case Analysis")
    print("=" * 60)

    selected = select_recordings()

    for record in selected:

        print("\n" + "-" * 50)
        print(f"Class: {record['class']}")
        print(f"Fold: {record['fold']}")
        print(f"Filename: {record['filename']}")

        audio = load_recording(record["path"])

        features = extract_features(audio)

        print(f"Audio samples: {len(audio)}")
        print(f"STFT shape: {features['stft_db'].shape}")
        print(f"MFCC shape: {features['mfcc'].shape}")
        print(f"Log-Mel shape: {features['logmel'].shape}")

        plot_audio_case(
            record,
            audio,
            features
        )

    print("\nAll real audio case analyses completed!")


if __name__ == "__main__":
    main()
