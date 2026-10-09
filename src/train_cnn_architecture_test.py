
"""
ELEC5305 Project
Environmental Sound Classification

Model B: Time-resolved MFCC + CNN
Model C: Log-Mel Spectrogram + CNN

This stage verifies the shared CNN architecture.
Synthetic tensors are used for functionality testing.

Author: Xiao Hu
"""

import torch
import torch.nn as nn


# ============================================================
# Section 1: Configuration
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_CLASSES = 10
BATCH_SIZE = 8

MFCC_SHAPE = (40, 173)
LOGMEL_SHAPE = (128, 173)


# ============================================================
# Section 2: Shared CNN Architecture
# ============================================================

class AudioCNN(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES):

        super().__init__()

        self.features = nn.Sequential(

            # Convolutional Block 1
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Convolutional Block 2
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Convolutional Block 3
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Convert varying spectrogram sizes to
            # a fixed-length representation
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


# ============================================================
# Section 3: Parameter Count
# ============================================================

def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ============================================================
# Section 4: Model Functionality Test
# ============================================================

def test_model(model_name, feature_shape):

    print("\n" + "=" * 55)
    print(f"Testing: {model_name}")
    print("=" * 55)

    # Fix seed for reproducibility
    torch.manual_seed(42)

    # Create model
    model = AudioCNN().to(DEVICE)

    # Input format:
    # (batch, channels, frequency_features, time_frames)
    inputs = torch.randn(
        BATCH_SIZE,
        1,
        feature_shape[0],
        feature_shape[1],
        device=DEVICE
    )

    targets = torch.randint(
        low=0,
        high=NUM_CLASSES,
        size=(BATCH_SIZE,),
        device=DEVICE
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # Forward propagation
    model.train()
    outputs = model(inputs)

    print(f"Input shape: {tuple(inputs.shape)}")
    print(f"Output shape: {tuple(outputs.shape)}")

    assert outputs.shape == (
        BATCH_SIZE,
        NUM_CLASSES
    ), "Unexpected CNN output dimensions."

    assert torch.isfinite(outputs).all(), (
        "CNN output contains NaN or infinity."
    )

    # Loss calculation
    loss = criterion(outputs, targets)

    print(f"Initial loss: {loss.item():.4f}")

    # Backward propagation
    optimizer.zero_grad()
    loss.backward()

    # Verify at least one gradient exists
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert all(
        grad is not None
        for grad in gradients
    ), "Missing gradients."

    assert all(
        torch.isfinite(grad).all()
        for grad in gradients
    ), "Invalid gradients."

    optimizer.step()

    print("Forward pass: PASSED")
    print("Backward pass: PASSED")
    print("Optimizer update: PASSED")

    parameter_count = count_parameters(model)

    print(f"Trainable parameters: {parameter_count:,}")

    return parameter_count


# ============================================================
# Section 5: Main
# ============================================================

def main():

    print("=" * 55)
    print("ELEC5305 - Shared CNN Architecture Verification")
    print("=" * 55)

    print(f"\nDevice: {DEVICE}")

    mfcc_parameters = test_model(
        "Model B - Time-resolved MFCC CNN",
        MFCC_SHAPE
    )

    logmel_parameters = test_model(
        "Model C - Log-Mel Spectrogram CNN",
        LOGMEL_SHAPE
    )

    # Verify identical trainable parameter counts
    assert mfcc_parameters == logmel_parameters

    print("\n" + "=" * 55)
    print("Architecture Comparison")
    print("=" * 55)

    print(f"MFCC-CNN parameters: {mfcc_parameters:,}")
    print(f"Log-Mel CNN parameters: {logmel_parameters:,}")

    print("\nBoth CNN models use identical architectures.")
    print("All architecture verification tests PASSED!")

    print(
        "\nNote: Random tensors were used. "
        "No UrbanSound8K classification was performed."
    )


if __name__ == "__main__":
    main()
