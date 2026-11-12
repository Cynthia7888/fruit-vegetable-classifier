"""Shared model construction and image inference for fruit/vegetable classifiers."""

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import models, transforms
from torch import nn


IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
MODEL_NAMES = ("custom_cnn", "resnet50", "vgg16")


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()

        def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(3, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(image))


def build_model(model_name: str, num_classes: int) -> nn.Module:
    """Build the same layer layouts used by the training notebook."""
    if model_name == "custom_cnn":
        return CustomCNN(num_classes)
    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(model.fc.in_features, num_classes)
        )
        return model
    if model_name == "vgg16":
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(4096, num_classes)
        )
        return model
    raise ValueError(f"Unknown model '{model_name}'. Choose from {', '.join(MODEL_NAMES)}.")


def load_classifier(
    model_name: str,
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> tuple[nn.Module, list[str], torch.device]:
    """Load a compatible checkpoint and return its model, labels, and device."""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {', '.join(MODEL_NAMES)}.")

    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_file}")

    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint: Any = torch.load(
            checkpoint_file, map_location=target_device, weights_only=True
        )
    except TypeError as exc:
        raise RuntimeError(
            "This application requires PyTorch with torch.load(weights_only=True) support."
        ) from exc

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid checkpoint format in {checkpoint_file}.")

    classes = checkpoint.get("classes")
    if not isinstance(classes, list) or not classes or not all(
        isinstance(label, str) for label in classes
    ):
        raise ValueError(f"Checkpoint {checkpoint_file} has no valid 'classes' list.")
    if checkpoint.get("model_name", model_name) != model_name:
        raise ValueError(
            f"Checkpoint is for '{checkpoint['model_name']}', not '{model_name}'."
        )

    state_dict = checkpoint.get("model_state")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint {checkpoint_file} has no valid 'model_state'.")

    model = build_model(model_name, len(classes))
    model.load_state_dict(state_dict)
    model.to(target_device)
    model.eval()
    return model, classes, target_device


def predict_image(
    model: nn.Module,
    image: Image.Image,
    classes: list[str],
    device: torch.device,
    top_k: int = 5,
) -> list[dict[str, float | str]]:
    """Return top-k class names and probabilities for an RGB or PIL image."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if not classes:
        raise ValueError("The class list must not be empty.")

    preprocess = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    batch = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(batch).softmax(dim=1)[0]

    count = min(top_k, len(classes))
    scores, indices = probabilities.topk(count)
    return [
        {"class": classes[index], "probability": float(score)}
        for score, index in zip(scores.cpu().tolist(), indices.cpu().tolist())
    ]
