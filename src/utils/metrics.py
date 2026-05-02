from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    correct = (predictions == targets).sum().item()
    return correct / max(targets.size(0), 1)

