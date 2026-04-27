"""Additional research analyses for report3 loss-landscape study."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from metrics import compute_metrics
from model import build_model
from quadratic_fit import fit_full_quadratic
from surface import _eval_loss_batches, _normalize_directions, _set_seeds


@dataclass
class AnalysisConfig:
    model_name: str = "model1"
    radius: float = 0.1
    grid_steps: int = 11
    max_batches: int = 2
    seed: int = 42
    radii: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5)


def _flatten_direction(d: dict[str, torch.Tensor], names: list[str]) -> torch.Tensor:
    return torch.cat([d[n].reshape(-1) for n in names])


def _unflatten_to_dict(
    flat: torch.Tensor, base: dict[str, torch.Tensor], names: list[str]
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    idx = 0
    for n in names:
        ref = base[n]
        numel = ref.numel()
        out[n] = flat[idx : idx + numel].view_as(ref).clone()
        idx += numel
    return out


def _layer_normalize_direction(
    theta_state: dict[str, torch.Tensor], direction: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, theta in theta_state.items():
        d = direction[name].clone()
        tn = theta.norm().clamp_min(1e-12)
        dn = d.norm().clamp_min(1e-12)
        out[name] = d * (tn / dn)
    return out


def _compute_surface_with_directions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    base_params: dict[str, torch.Tensor],
    full_base_state: dict[str, torch.Tensor],
    d1: dict[str, torch.Tensor],
    d2: dict[str, torch.Tensor],
    radius: float,
    steps: int,
    max_batches: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alphas = np.linspace(-radius, radius, steps, dtype=np.float64)
    betas = np.linspace(-radius, radius, steps, dtype=np.float64)
    alpha_grid, beta_grid = np.meshgrid(alphas, betas, indexing="ij")
    f = np.zeros((steps, steps), dtype=np.float64)

    model = model.to(device)
    for i in range(steps):
        for j in range(steps):
            a = float(alpha_grid[i, j])
            b = float(beta_grid[i, j])
            shifted_params = {
                name: (base_params[name] + a * d1[name] + b * d2[name]).to(device)
                for name in base_params
            }
            current_state = {k: v.to(device) for k, v in full_base_state.items()}
            current_state.update(shifted_params)
            model.load_state_dict(current_state, strict=True)
            f[i, j] = _eval_loss_batches(model, loader, device, max_batches=max_batches)

    restore_state = {k: v.to(device) for k, v in full_base_state.items()}
    model.load_state_dict(restore_state, strict=True)
    return f, alpha_grid, beta_grid


def _relative_l2_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.linalg.norm(y_true)
    if denom <= 1e-12:
        return float("nan")
    return float(np.linalg.norm(y_true - y_pred) / denom)


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _plot_true_vs_quadratic(
    alpha: np.ndarray, beta: np.ndarray, f_true: np.ndarray, f_quad: np.ndarray, out_path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    levels = 20
    axes[0].contourf(alpha, beta, f_true, levels=levels, cmap="viridis")
    axes[0].set_title("Real surface")
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("beta")
    axes[1].contourf(alpha, beta, f_quad, levels=levels, cmap="viridis")
    axes[1].set_title("Full quadratic")
    axes[1].set_xlabel("alpha")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_error_vs_radius(df: pd.DataFrame, out_linear: Path, out_loglog: Path) -> float:
    plt.figure(figsize=(6, 4))
    plt.plot(df["radius"], df["RMSE"], marker="o", label="RMSE")
    plt.plot(df["radius"], df["MAE"], marker="s", label="MAE")
    plt.xlabel("radius")
    plt.ylabel("error")
    plt.title("Quadratic approximation error vs radius")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_linear, dpi=160)
    plt.close()

    x = np.log(df["radius"].to_numpy())
    y = np.log(df["RMSE"].to_numpy())
    p, log_c = np.polyfit(x, y, deg=1)
    c = math.exp(log_c)

    plt.figure(figsize=(6, 4))
    plt.loglog(df["radius"], df["RMSE"], "o-", label="RMSE")
    fit = c * np.power(df["radius"].to_numpy(), p)
    plt.loglog(df["radius"], fit, "--", label=f"fit: C*r^p, p={p:.3f}")
    plt.xlabel("radius (log)")
    plt.ylabel("RMSE (log)")
    plt.title("Log-log error scaling")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_loglog, dpi=160)
    plt.close()
    return float(p)


def _load_checkpoint_state(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    return ckpt


def run_additional_analyses(
    cfg: AnalysisConfig,
    loader: DataLoader,
    device: torch.device,
    checkpoint_dir: Path,
    figures_dir: Path,
    tables_dir: Path,
) -> dict[str, Any]:
    _set_seeds(cfg.seed)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    final_ckpt = checkpoint_dir / f"{cfg.model_name}_final.pth"
    if not final_ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {final_ckpt}")

    model = build_model(cfg.model_name).to(device)
    ckpt = _load_checkpoint_state(final_ckpt, device)
    model.load_state_dict(ckpt["state_dict"])

    param_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    names = [n for n, _ in model.named_parameters()]
    full_base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # ------- 1) Direction sensitivity -------
    direction_rows: list[dict[str, Any]] = []

    raw1 = {n: torch.randn_like(t) for n, t in param_state.items()}
    raw2 = {n: torch.randn_like(t) for n, t in param_state.items()}

    d1_filter, d2_filter = _normalize_directions(param_state, raw1, raw2)
    direction_variants: dict[str, tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]] = {
        "random_filter_normalized": (d1_filter, d2_filter),
        "layer_normalized": (
            _layer_normalize_direction(param_state, raw1),
            _layer_normalize_direction(param_state, raw2),
        ),
    }

    # gradient-aligned
    model.zero_grad(set_to_none=True)
    for p in model.parameters():
        p.requires_grad_(True)
    criterion = torch.nn.CrossEntropyLoss()
    loss_sum = 0.0
    n_sum = 0
    for i, (images, targets) in enumerate(loader):
        if i >= cfg.max_batches:
            break
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss_sum += criterion(logits, targets) * targets.numel()
        n_sum += targets.numel()
    (loss_sum / max(n_sum, 1)).backward()
    grad_flat = torch.cat([p.grad.detach().reshape(-1).cpu() for p in model.parameters()])
    for p in model.parameters():
        p.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    grad_flat = grad_flat / grad_flat.norm().clamp_min(1e-12)

    rnd_flat = _flatten_direction(d1_filter, names)
    proj = torch.dot(rnd_flat, grad_flat) / torch.dot(grad_flat, grad_flat)
    ortho = rnd_flat - proj * grad_flat
    ortho = ortho / ortho.norm().clamp_min(1e-12)
    d1_grad = _unflatten_to_dict(grad_flat, param_state, names)
    d2_grad = _unflatten_to_dict(ortho, param_state, names)
    direction_variants["gradient_aligned"] = (d1_grad, d2_grad)

    # trajectory direction if possible
    traj_ckpts = sorted(checkpoint_dir.glob(f"{cfg.model_name}_epoch*.pth"))
    if traj_ckpts:
        near = _load_checkpoint_state(traj_ckpts[-1], device)["state_dict"]
        final_state = ckpt["state_dict"]
        d1_traj_parts = []
        for n, p in model.named_parameters():
            d1_traj_parts.append((final_state[n] - near[n]).detach().cpu().reshape(-1))
        d1_traj_flat = torch.cat(d1_traj_parts)
        if d1_traj_flat.norm() > 1e-12:
            d1_traj_flat = d1_traj_flat / d1_traj_flat.norm()
            d2_traj_flat = rnd_flat - torch.dot(rnd_flat, d1_traj_flat) * d1_traj_flat
            d2_traj_flat = d2_traj_flat / d2_traj_flat.norm().clamp_min(1e-12)
            direction_variants["trajectory"] = (
                _unflatten_to_dict(d1_traj_flat, param_state, names),
                _unflatten_to_dict(d2_traj_flat, param_state, names),
            )
        else:
            direction_rows.append(
                {
                    "direction_type": "trajectory",
                    "status": "todo_zero_direction",
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "RelRMSE": np.nan,
                    "relative_l2_error": np.nan,
                }
            )
    else:
        direction_rows.append(
            {
                "direction_type": "trajectory",
                "status": "todo_missing_epoch_checkpoints",
                "MAE": np.nan,
                "RMSE": np.nan,
                "RelRMSE": np.nan,
                "relative_l2_error": np.nan,
            }
        )

    for direction_type, (d1, d2) in direction_variants.items():
        f_true, alpha, beta = _compute_surface_with_directions(
            model=model,
            loader=loader,
            device=device,
            base_params=param_state,
            full_base_state=full_base_state,
            d1=d1,
            d2=d2,
            radius=cfg.radius,
            steps=cfg.grid_steps,
            max_batches=cfg.max_batches,
        )
        _, f_quad = fit_full_quadratic(alpha, beta, f_true)
        m = compute_metrics(f_true, f_quad)
        row = {
            "direction_type": direction_type,
            "status": "ok",
            "MAE": _mae(f_true, f_quad),
            "RMSE": m["RMSE"],
            "RelRMSE": m["RelRMSE"],
            "relative_l2_error": _relative_l2_error(f_true, f_quad),
        }
        direction_rows.append(row)
        _plot_true_vs_quadratic(
            alpha,
            beta,
            f_true,
            f_quad,
            figures_dir / f"direction_{direction_type}_comparison.png",
        )

    df_direction = pd.DataFrame(direction_rows)
    direction_csv = tables_dir / "direction_sensitivity.csv"
    df_direction.to_csv(direction_csv, index=False)

    # ------- 2) Error scaling by radius -------
    error_rows: list[dict[str, Any]] = []
    # fixed directions for fair scaling
    fixed_d1, fixed_d2 = d1_filter, d2_filter
    for r in cfg.radii:
        f_true, alpha, beta = _compute_surface_with_directions(
            model=model,
            loader=loader,
            device=device,
            base_params=param_state,
            full_base_state=full_base_state,
            d1=fixed_d1,
            d2=fixed_d2,
            radius=float(r),
            steps=cfg.grid_steps,
            max_batches=cfg.max_batches,
        )
        params, f_quad = fit_full_quadratic(alpha, beta, f_true)
        m = compute_metrics(f_true, f_quad)
        error_rows.append(
            {
                "radius": r,
                "MAE": _mae(f_true, f_quad),
                "RMSE": m["RMSE"],
                "RelRMSE": m["RelRMSE"],
                "relative_l2_error": _relative_l2_error(f_true, f_quad),
                "h11": params["a"],
                "h12": params["b"],
                "h22": params["d"],
            }
        )
    df_error = pd.DataFrame(error_rows)
    p_hat = _plot_error_vs_radius(
        df_error,
        figures_dir / "error_vs_radius.png",
        figures_dir / "error_vs_radius_loglog.png",
    )
    df_error["power_law_p"] = p_hat
    error_csv = tables_dir / "error_by_radius.csv"
    df_error.to_csv(error_csv, index=False)

    # ------- 3) Training dynamics: flat vs sharp -------
    dyn_rows: list[dict[str, Any]] = []
    candidates = [
        ("early", checkpoint_dir / f"{cfg.model_name}_epoch1.pth"),
        ("middle", checkpoint_dir / f"{cfg.model_name}_epoch5.pth"),
        ("final", final_ckpt),
    ]

    for label, path in candidates:
        if not path.exists():
            dyn_rows.append(
                {
                    "checkpoint_label": label,
                    "checkpoint_file": path.name,
                    "status": "todo_missing_checkpoint",
                    "lambda_min": np.nan,
                    "lambda_max": np.nan,
                    "condition_number": np.nan,
                    "sharpness": np.nan,
                }
            )
            continue
        c = _load_checkpoint_state(path, device)
        model.load_state_dict(c["state_dict"])
        cur_params = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
        cur_full_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        f_true, alpha, beta = _compute_surface_with_directions(
            model=model,
            loader=loader,
            device=device,
            base_params=cur_params,
            full_base_state=cur_full_state,
            d1=fixed_d1,
            d2=fixed_d2,
            radius=cfg.radius,
            steps=cfg.grid_steps,
            max_batches=cfg.max_batches,
        )
        params, _ = fit_full_quadratic(alpha, beta, f_true)
        H = np.array([[params["a"], params["b"]], [params["b"], params["d"]]], dtype=np.float64)
        evals = np.linalg.eigvalsh(H)
        lam_min = float(np.min(evals))
        lam_max = float(np.max(evals))
        cond = float(lam_max / lam_min) if lam_min > 1e-12 else float("inf")
        center_idx = cfg.grid_steps // 2
        sharpness = float(np.max(f_true) - f_true[center_idx, center_idx])
        dyn_rows.append(
            {
                "checkpoint_label": label,
                "checkpoint_file": path.name,
                "status": "ok",
                "lambda_min": lam_min,
                "lambda_max": lam_max,
                "condition_number": cond,
                "sharpness": sharpness,
            }
        )
        plt.figure(figsize=(6, 5))
        plt.contourf(alpha, beta, f_true, levels=20, cmap="viridis")
        plt.xlabel("alpha")
        plt.ylabel("beta")
        plt.title(f"Loss surface ({label})")
        plt.tight_layout()
        plt.savefig(figures_dir / f"loss_surface_epoch_{label}.png", dpi=150)
        plt.close()

    df_dyn = pd.DataFrame(dyn_rows)
    dyn_csv = tables_dir / "training_dynamics.csv"
    df_dyn.to_csv(dyn_csv, index=False)

    ok_dyn = df_dyn[df_dyn["status"] == "ok"]
    if not ok_dyn.empty:
        plt.figure(figsize=(7, 4))
        x = np.arange(len(ok_dyn))
        labels = ok_dyn["checkpoint_label"].tolist()
        plt.plot(x, ok_dyn["sharpness"], marker="o", label="sharpness")
        plt.plot(x, ok_dyn["condition_number"], marker="s", label="condition_number")
        plt.xticks(x, labels)
        plt.title("Curvature and sharpness dynamics")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "curvature_dynamics.png", dpi=150)
        plt.close()

    summary = {
        "direction_csv": str(direction_csv),
        "error_csv": str(error_csv),
        "dynamics_csv": str(dyn_csv),
        "power_law_p": p_hat,
    }
    print("=== Additional analyses summary ===")
    print(pd.DataFrame([summary]).to_string(index=False))
    return summary
