from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.fer2013_dataset import build_dataloaders
from src.engine.trainer import evaluate, fit
from src.models.factory import build_model
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_training_curves,
    save_classification_report,
    save_history_csv,
)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            reduction="none"
        )

        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FER models on FER2013.")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/train_run")
    parser.add_argument("--model", type=str, default="resnet_cbam", choices=["cnn_cbam", "resnet_cbam"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_run_config(args: argparse.Namespace, class_names: list[str], output_dir: Path) -> None:
    config = vars(args).copy()
    config["class_names"] = class_names
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in the current environment.")
        return torch.device("cuda")

    if device_name == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available in the current environment.")
        return torch.device("mps")

    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    report_dir = output_dir / "reports"
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    save_run_config(args, bundle.class_names, output_dir)

    model = build_model(args.model, num_classes=len(bundle.class_names)).to(device)
    criterion = FocalLoss(
        alpha=None,
        gamma=1.0,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    history = fit(
        model=model,
        train_loader=bundle.train_loader,
        val_loader=bundle.val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        checkpoint_path=checkpoint_dir / "best_model.pth",
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        early_stopping_patience=args.early_stopping_patience,
    )

    model.load_state_dict(torch.load(checkpoint_dir / "best_model.pth", map_location=device))
    test_result = evaluate(
        model,
        bundle.test_loader,
        criterion,
        device,
        max_batches=args.max_test_batches,
        stage="test",
    )

    save_history_csv(history, report_dir)
    plot_training_curves(history, plot_dir)
    plot_confusion_matrix(test_result.targets, test_result.predictions, bundle.class_names, plot_dir)
    save_classification_report(test_result.targets, test_result.predictions, bundle.class_names, report_dir)

    summary = {
        "model": args.model,
        "loss": "FocalLoss",
        "gamma": 1.0,
        "alpha": "none",
        "device": str(device),
        "epochs_ran": len(history),
        "best_epoch": max(history, key=lambda item: item["val_acc"])["epoch"],
        "best_val_acc": max(item["val_acc"] for item in history),
        "final_test_loss": test_result.metrics.loss,
        "final_test_acc": test_result.metrics.accuracy,
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
