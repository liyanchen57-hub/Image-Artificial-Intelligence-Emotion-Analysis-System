from __future__ import annotations

from torch import nn

from src.models.cnn_cbam import EmotionCNNCBAM
from src.models.resnet_cbam import EmotionResNetCBAM


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "cnn_cbam":
        return EmotionCNNCBAM(num_classes=num_classes)
    if model_name == "resnet_cbam":
        return EmotionResNetCBAM(num_classes=num_classes)
    raise ValueError(f"Unsupported model: {model_name}")
