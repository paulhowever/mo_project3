"""Обучение модели на CIFAR-10: аугментации, SGD, CosineAnnealingLR, чекпоинты."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config  # noqa: E402
from model import build_model  # noqa: E402


def _set_seeds() -> None:
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)


def _ensure_dirs() -> None:
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)


def get_train_val_loaders():
    """Train с аугментациями и val без них (нормализация CIFAR-10)."""
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    train_set = datasets.CIFAR10(
        root=config.DATA_DIR, train=True, download=True, transform=train_tf
    )
    val_set = datasets.CIFAR10(
        root=config.DATA_DIR, train=False, download=True, transform=val_tf
    )
    worker_count = 0 if sys.platform == "darwin" else 2
    train_loader = DataLoader(
        train_set,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=worker_count,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=worker_count,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Доля верных ответов на выборке."""
    model.eval()
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        pred = logits.argmax(dim=1)
        correct += (pred == targets).sum().item()
        total += targets.numel()
    return correct / max(total, 1)


def train_model(model: nn.Module, model_name: str, cfg) -> None:
    """
    Обучение: SGD(momentum, weight_decay), CosineAnnealingLR, чекпоинты каждые 10 эпох
    и финальный CHECKPOINT_DIR/{model_name}_final.pth.
    """
    device = torch.device(cfg.DEVICE)
    model = model.to(device)
    train_loader, val_loader = get_train_val_loaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=cfg.LR,
        momentum=0.9,
        weight_decay=5e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS}", leave=False)
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=loss.item())
        scheduler.step()
        train_loss = running_loss / max(n_batches, 1)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")

        state = {
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": val_acc,
        }
        if epoch % 10 == 0 or epoch == cfg.EPOCHS:
            path = os.path.join(cfg.CHECKPOINT_DIR, f"{model_name}_epoch{epoch}.pth")
            torch.save(state, path)

    final_path = os.path.join(cfg.CHECKPOINT_DIR, f"{model_name}_final.pth")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "epoch": cfg.EPOCHS,
            "train_loss": train_loss,
            "val_accuracy": val_acc,
        },
        final_path,
    )
    print(f"Финальная val accuracy: {val_acc:.4f}")


def main() -> None:
    _set_seeds()
    _ensure_dirs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["model1", "model2"])
    args = parser.parse_args()
    model = build_model(args.model)
    train_model(model, args.model, config)


if __name__ == "__main__":
    main()
