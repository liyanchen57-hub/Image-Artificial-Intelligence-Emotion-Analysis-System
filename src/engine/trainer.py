from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.utils.metrics import EpochMetrics


@dataclass
class EvalResult:
    metrics: EpochMetrics
    predictions: list[int]
    targets: list[int]


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    max_batches: int | None = None,
    epoch: int | None = None,
    total_epochs: int | None = None,
) -> EpochMetrics:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_batches = len(dataloader) if max_batches is None else min(len(dataloader), max_batches)
    progress = tqdm(
        dataloader,
        total=total_batches,
        desc=_build_progress_desc("train", epoch, total_epochs),
        dynamic_ncols=True,
        leave=False,
    )

    for batch_index, (images, targets) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += targets.size(0)
        progress.set_postfix(
            loss=f"{total_loss / max(total_samples, 1):.4f}",
            acc=f"{total_correct / max(total_samples, 1):.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

        if max_batches is not None and batch_index >= max_batches:
            break

    progress.close()

    return EpochMetrics(
        loss=total_loss / max(total_samples, 1),
        accuracy=total_correct / max(total_samples, 1),
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
    epoch: int | None = None,
    total_epochs: int | None = None,
    stage: str = "eval",
) -> EvalResult:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_predictions: list[int] = []
    all_targets: list[int] = []
    total_batches = len(dataloader) if max_batches is None else min(len(dataloader), max_batches)
    progress = tqdm(
        dataloader,
        total=total_batches,
        desc=_build_progress_desc(stage, epoch, total_epochs),
        dynamic_ncols=True,
        leave=False,
    )

    for batch_index, (images, targets) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        predictions = logits.argmax(dim=1)
        total_loss += loss.item() * targets.size(0)
        total_correct += (predictions == targets).sum().item()
        total_samples += targets.size(0)
        all_predictions.extend(predictions.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())
        progress.set_postfix(
            loss=f"{total_loss / max(total_samples, 1):.4f}",
            acc=f"{total_correct / max(total_samples, 1):.4f}",
        )

        if max_batches is not None and batch_index >= max_batches:
            break

    progress.close()

    return EvalResult(
        metrics=EpochMetrics(
            loss=total_loss / max(total_samples, 1),
            accuracy=total_correct / max(total_samples, 1),
        ),
        predictions=all_predictions,
        targets=all_targets,
    )


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: ReduceLROnPlateau | None,
    device: torch.device,
    epochs: int,
    checkpoint_path: str | Path,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    early_stopping_patience: int | None = None,
) -> list[dict[str, float]]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_batches=max_train_batches,
            epoch=epoch,
            total_epochs=epochs,
        )
        val_result = evaluate(
            model,
            val_loader,
            criterion,
            device,
            max_batches=max_val_batches,
            epoch=epoch,
            total_epochs=epochs,
            stage="val",
        )
        epoch_seconds = time.perf_counter() - epoch_start

        if scheduler is not None:
            scheduler.step(val_result.metrics.loss)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                "train_acc": train_metrics.accuracy,
                "val_loss": val_result.metrics.loss,
                "val_acc": val_result.metrics.accuracy,
            }
        )

        if val_result.metrics.accuracy >= best_val_acc:
            best_val_acc = val_result.metrics.accuracy
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"time={epoch_seconds:.1f}s | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.4f} | "
            f"val_loss={val_result.metrics.loss:.4f} val_acc={val_result.metrics.accuracy:.4f}",
            flush=True,
        )

        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch:03d} | best_epoch={best_epoch:03d} best_val_acc={best_val_acc:.4f}",
                flush=True,
            )
            break

    return history


def _build_progress_desc(stage: str, epoch: int | None, total_epochs: int | None) -> str:
    if epoch is None or total_epochs is None:
        return stage
    return f"{stage} {epoch:03d}/{total_epochs:03d}"
