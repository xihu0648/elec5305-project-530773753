
"""
ELEC5305 Project
Shared Experimental Configuration

Loads and validates project settings from config.json.
"""

import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_DIR / "config.json"


def load_config():
    """Load and validate the shared configuration."""

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_PATH}"
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    required_sections = [
        "project",
        "audio",
        "training",
        "evaluation",
        "models"
    ]

    for section in required_sections:
        if section not in config:
            raise ValueError(
                f"Missing configuration section: {section}"
            )

    training = config["training"]
    audio = config["audio"]

    assert training["batch_size"] > 0
    assert training["learning_rate"] > 0
    assert training["max_epochs"] > 0
    assert training["early_stopping_patience"] > 0

    assert audio["sample_rate"] > 0
    assert audio["n_fft"] > 0
    assert audio["hop_length"] > 0

    assert config["evaluation"]["test_folds"] == list(
        range(1, 11)
    )

    return config


def main():

    config = load_config()

    print("=" * 55)
    print("ELEC5305 - Shared Configuration Verification")
    print("=" * 55)

    print(f"\nProject: {config['project']['title']}")

    print("\nTraining Parameters:")

    for key, value in config["training"].items():
        print(f"  {key}: {value}")

    print("\nAudio Parameters:")

    for key, value in config["audio"].items():
        print(f"  {key}: {value}")

    print("\nConfiguration validation: PASSED")


if __name__ == "__main__":
    main()
