"""Вычисление двумерного сечения loss landscape (Li et al., 2018)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config  # noqa: E402
from model import build_model  # noqa: E402


def _set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_dirs() -> None:
    os.makedirs(config.RESULTS_DIR, exist_ok=True)


def get_val_loader():
    """CIFAR-10 test/val без аугментаций."""
    from torchvision import datasets, transforms

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    val_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    worker_count = 0 if sys.platform == "darwin" else 2
    val_set = datasets.CIFAR10(
        root=config.DATA_DIR, train=False, download=True, transform=val_tf
    )
    return DataLoader(
        val_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=worker_count,
        pin_memory=torch.cuda.is_available(),
    )


def _random_direction_like_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, t in state.items():
        out[name] = torch.randn_like(t, dtype=t.dtype)
    return out


def _filterwise_normalize_direction(
    theta: torch.Tensor, direction: torch.Tensor
) -> torch.Tensor:
    """
    Для Conv2d веса [out, in, k, k]: нормировать каждый выходной фильтр под норму θ.
    Иначе — масштабировать весь тензор на ||θ||/||d||.
    """
    d = direction.clone()
    if theta.dim() == 4:
        for i in range(theta.shape[0]):
            tn = theta[i].flatten().norm()
            dn = d[i].flatten().norm().clamp_min(1e-12)
            d[i] = d[i] * (tn / dn)
    else:
        tn = theta.norm()
        dn = d.norm().clamp_min(1e-12)
        d = d * (tn / dn)
    return d


def _normalize_directions(
    base_state: dict[str, torch.Tensor],
    d1: dict[str, torch.Tensor],
    d2: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    d1n: dict[str, torch.Tensor] = {}
    d2n: dict[str, torch.Tensor] = {}
    for name in base_state:
        th = base_state[name]
        d1n[name] = _filterwise_normalize_direction(th, d1[name])
        d2n[name] = _filterwise_normalize_direction(th, d2[name])
    return d1n, d2n


def _gram_schmidt_orthogonalize(
    d1: dict[str, torch.Tensor],
    d2: dict[str, torch.Tensor],
    names: list[str],
) -> dict[str, torch.Tensor]:
    """
    Ортогонализация d2 относительно d1 (Грам-Шмидт) в пространстве параметров.
    После ортогонализации d2 перенормируется filter-wise.
    Гарантирует <d1_flat, d2_flat> ≈ 0.
    """
    d1_flat = torch.cat([d1[n].reshape(-1) for n in names])
    d2_flat = torch.cat([d2[n].reshape(-1) for n in names])
    proj = (d2_flat @ d1_flat) / (d1_flat @ d1_flat).clamp_min(1e-12)
    d2_flat_orth = d2_flat - proj * d1_flat
    idx = 0
    d2_orth: dict[str, torch.Tensor] = {}
    for n in names:
        sz = d1[n].numel()
        d2_orth[n] = d2_flat_orth[idx : idx + sz].reshape(d1[n].shape)
        idx += sz
    return d2_orth


def _state_from_base_and_directions(
    base_state: dict[str, torch.Tensor],
    d1: dict[str, torch.Tensor],
    d2: dict[str, torch.Tensor],
    alpha: float,
    beta: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    new_state: dict[str, torch.Tensor] = {}
    for name, t in base_state.items():
        new_state[name] = (t + alpha * d1[name] + beta * d2[name]).to(device)
    return new_state


@torch.no_grad()
def _eval_loss_batches(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 10,
) -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_n = 0
    for i, (images, targets) in enumerate(loader):
        if i >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item()
        total_n += targets.numel()
    return total_loss / max(total_n, 1)


def _flatten_state_dict(d: dict[str, torch.Tensor], names: list[str]) -> torch.Tensor:
    return torch.cat([d[n].reshape(-1) for n in names])


def make_random_filterwise_directions(
    model: nn.Module, seed: int
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[str]]:
    """
    Воспроизводимо построить пару (d1, d2): filter-wise нормированных и
    ортогонализированных направлений. Тот же seed → те же направления, что использует
    compute_surface, поэтому полученные d1/d2 пригодны для повторного использования
    (например, для перерасчёта поверхности в собственном базисе Гессиана).
    """
    _set_seeds(seed)
    names = [n for n, _ in model.named_parameters()]
    param_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    raw1 = _random_direction_like_state(param_state)
    raw2 = _random_direction_like_state(param_state)
    d1, d2 = _normalize_directions(param_state, raw1, raw2)
    d2 = _gram_schmidt_orthogonalize(d1, d2, names)
    d2_renorm: dict[str, torch.Tensor] = {}
    for n in names:
        d2_renorm[n] = _filterwise_normalize_direction(param_state[n], d2[n])
    return d1, d2_renorm, names


def _scan_grid_with_directions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    param_state: dict[str, torch.Tensor],
    full_base_state: dict[str, torch.Tensor],
    d1: dict[str, torch.Tensor],
    d2: dict[str, torch.Tensor],
    radius: float,
    steps: int,
    max_batches: int,
    desc: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alphas = np.linspace(-radius, radius, steps, dtype=np.float64)
    betas = np.linspace(-radius, radius, steps, dtype=np.float64)
    alpha_grid, beta_grid = np.meshgrid(alphas, betas, indexing="ij")
    f = np.zeros((steps, steps), dtype=np.float64)
    model = model.to(device)
    total = steps * steps
    pbar = tqdm(range(total), desc=desc)
    for idx in pbar:
        i = idx // steps
        j = idx % steps
        a = float(alpha_grid[i, j])
        b = float(beta_grid[i, j])
        shifted_params = _state_from_base_and_directions(param_state, d1, d2, a, b, device)
        current_state = {k: v.to(device) for k, v in full_base_state.items()}
        current_state.update(shifted_params)
        model.load_state_dict(current_state, strict=True)
        with torch.no_grad():
            loss = _eval_loss_batches(model, loader, device, max_batches=max_batches)
        f[i, j] = loss
    return f, alpha_grid, beta_grid


def compute_surface(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    radius: float,
    steps: int,
    seed: int,
    model_name: str,
    cfg,
    max_batches: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Сечение f(α,β) = L(θ + α d1 + β d2) на сетке steps×steps.
    Сохраняет .npz с theta_flat, d1_flat, d2_flat для последующего q3 и метрик.
    При фиксированном seed направления d1/d2 воспроизводимы; при повторных вызовах
    с тем же seed (например, для разных радиусов) используется одна и та же плоскость.

    Направления d1 и d2 ортогонализированы (Грам-Шмидт) и нормированы filter-wise,
    что гарантирует геометрически чистые оси плоскости сечения.
    """
    _ensure_dirs()
    d1, d2, names = make_random_filterwise_directions(model, seed)
    param_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    full_base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    theta_flat = _flatten_state_dict(param_state, names)
    d1_flat = _flatten_state_dict(d1, names)
    d2_flat = _flatten_state_dict(d2, names)

    f, alpha_grid, beta_grid = _scan_grid_with_directions(
        model, loader, device,
        param_state=param_state,
        full_base_state=full_base_state,
        d1=d1, d2=d2,
        radius=radius, steps=steps,
        max_batches=max_batches,
        desc="Loss surface",
    )

    out_path = os.path.join(cfg.RESULTS_DIR, f"surface_{model_name}_r{radius}.npz")
    np.savez(
        out_path,
        alpha_grid=alpha_grid,
        beta_grid=beta_grid,
        f=f,
        theta_flat=theta_flat.numpy(),
        d1_flat=d1_flat.numpy(),
        d2_flat=d2_flat.numpy(),
        seed=np.array(seed),
    )
    restored_state = {k: v.to(device) for k, v in full_base_state.items()}
    model.load_state_dict(restored_state, strict=True)
    return f, alpha_grid, beta_grid


def compute_surface_along_directions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    radius: float,
    steps: int,
    model_name: str,
    cfg,
    d1: dict[str, torch.Tensor],
    d2: dict[str, torch.Tensor],
    out_filename: str,
    max_batches: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Аналогично compute_surface, но направления d1, d2 задаются извне (например,
    собственные векторы 2×2-проекции Гессиана внутри уже выбранной плоскости).
    Имя выходного .npz задаётся параметром out_filename, чтобы не перезаписывать
    основной кэш surface_{model}_r{r}.npz.
    """
    _ensure_dirs()
    names = [n for n, _ in model.named_parameters()]
    param_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    full_base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    theta_flat = _flatten_state_dict(param_state, names)
    d1_flat = _flatten_state_dict(d1, names)
    d2_flat = _flatten_state_dict(d2, names)

    f, alpha_grid, beta_grid = _scan_grid_with_directions(
        model, loader, device,
        param_state=param_state,
        full_base_state=full_base_state,
        d1=d1, d2=d2,
        radius=radius, steps=steps,
        max_batches=max_batches,
        desc=f"Surface ({out_filename})",
    )

    out_path = os.path.join(cfg.RESULTS_DIR, out_filename)
    np.savez(
        out_path,
        alpha_grid=alpha_grid,
        beta_grid=beta_grid,
        f=f,
        theta_flat=theta_flat.numpy(),
        d1_flat=d1_flat.numpy(),
        d2_flat=d2_flat.numpy(),
    )
    restored_state = {k: v.to(device) for k, v in full_base_state.items()}
    model.load_state_dict(restored_state, strict=True)
    return f, alpha_grid, beta_grid


def main() -> None:
    _set_seeds(config.SEED)
    np.random.seed(config.SEED)
    _ensure_dirs()
    os.makedirs(config.DATA_DIR, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["model1", "model2"])
    parser.add_argument("--radius", type=float, default=config.RADIUS_DEFAULT)
    args = parser.parse_args()
    device = torch.device(config.DEVICE)
    model = build_model(args.model).to(device)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{args.model}_final.pth")
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    loader = get_val_loader()
    compute_surface(
        model,
        loader,
        device,
        radius=args.radius,
        steps=config.GRID_STEPS,
        seed=config.SEED,
        model_name=args.model,
        cfg=config,
    )


if __name__ == "__main__":
    main()
