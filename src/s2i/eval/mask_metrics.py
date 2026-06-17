"""Metric helpers for capability-relative token mask training."""

from __future__ import annotations

import torch


def masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weights = valid.float()
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def pearson_corr(x: torch.Tensor, y: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid.bool()
    if mask.sum() < 2:
        return torch.zeros((), device=x.device)
    x_valid = x[mask].float()
    y_valid = y[mask].float()
    x_centered = x_valid - x_valid.mean()
    y_centered = y_valid - y_valid.mean()
    denom = x_centered.norm() * y_centered.norm()
    if denom.item() == 0.0:
        return torch.zeros((), device=x.device)
    return (x_centered * y_centered).sum() / denom


def mask_summary(
    mask: torch.Tensor,
    valid: torch.Tensor,
    token_loss: torch.Tensor,
    entropy: torch.Tensor,
    margin: torch.Tensor,
) -> dict[str, torch.Tensor]:
    valid_f = valid.float()
    valid_count = valid_f.sum().clamp_min(1.0)
    mask_rate = (mask * valid_f).sum() / valid_count
    mask_clamped = mask.clamp(1e-6, 1.0 - 1e-6)
    mask_entropy = -(
        mask_clamped * mask_clamped.log()
        + (1.0 - mask_clamped) * (1.0 - mask_clamped).log()
    )
    return {
        "mask_rate": mask_rate,
        "mask_std": mask[valid.bool()].float().std(unbiased=False)
        if valid.bool().any()
        else torch.zeros((), device=mask.device),
        "mask_min": mask[valid.bool()].min()
        if valid.bool().any()
        else torch.zeros((), device=mask.device),
        "mask_max": mask[valid.bool()].max()
        if valid.bool().any()
        else torch.zeros((), device=mask.device),
        "mask_entropy": masked_mean(mask_entropy, valid),
        "valid_tokens": valid_f.sum(),
        "loss_mask_corr": pearson_corr(token_loss.detach(), mask.detach(), valid),
        "entropy_mask_corr": pearson_corr(entropy.detach(), mask.detach(), valid),
        "margin_mask_corr": pearson_corr(margin.detach(), mask.detach(), valid),
    }
