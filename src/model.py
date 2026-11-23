import torch.nn as nn
from torchvision import models

from src.config import NUM_CLASSES, PRETRAINED


def build_model(num_classes=NUM_CLASSES, pretrained=PRETRAINED):
    """
    Build a ResNet18 classifier for 12-class solar thermal defect detection.

    Args:
        num_classes: Number of defect categories.
        pretrained: Whether to use ImageNet-pretrained weights.

    Returns:
        ResNet18 model with a custom classification head.
    """

    weights = (
        models.ResNet18_Weights.DEFAULT
        if pretrained
        else None
    )

    model = models.resnet18(weights=weights)

    # Replace the original 1000-class ImageNet output layer.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
