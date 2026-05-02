from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]
    class_weights: torch.Tensor


def build_transforms(image_size: int = 48) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(
                degrees=12,
                translate=(0.08, 0.08),
                scale=(0.95, 1.05),
                shear=8,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.RandomErasing(
                p=0.25,
                scale=(0.02, 0.12),
                ratio=(0.3, 3.3),
                value="random",
            ),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    return train_transform, eval_transform


def _compute_class_weights(targets: list[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(targets, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.clip(counts, a_min=1.0, a_max=None)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(
    data_root: str | Path,
    batch_size: int = 64,
    num_workers: int = 0,
    image_size: int = 48,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> DataBundle:
    data_root = Path(data_root)
    train_dir = data_root / "train"
    test_dir = data_root / "test"

    train_transform, eval_transform = build_transforms(image_size=image_size)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    train_eval_dataset = datasets.ImageFolder(train_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    indices = np.arange(len(train_dataset))
    targets = np.array(train_dataset.targets)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_indices, val_indices = next(splitter.split(indices, targets))

    train_subset = Subset(train_dataset, train_indices.tolist())
    val_subset = Subset(train_eval_dataset, val_indices.tolist())

    train_targets = targets[train_indices].tolist()
    class_weights = _compute_class_weights(train_targets, num_classes=len(train_dataset.classes))

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=train_dataset.classes,
        class_weights=class_weights,
    )
