"""Capability-relative token mask components for Phase 4."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Iterator

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MaskHeadConfig:
    hidden_size: int
    hidden_proj: int = 128
    scalar_features: int = 4
    mlp_hidden: int = 256
    temperature: float = 1.0


class TokenMaskHead(nn.Module):
    """Lightweight token-level gate reading current-model states."""

    def __init__(self, config: MaskHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_proj = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_proj),
            nn.GELU(),
        )
        self.scalar_proj = nn.Sequential(
            nn.LayerNorm(config.scalar_features),
            nn.Linear(config.scalar_features, config.hidden_proj),
            nn.GELU(),
        )
        self.net = nn.Sequential(
            nn.Linear(config.hidden_proj * 2, config.mlp_hidden),
            nn.GELU(),
            nn.Linear(config.mlp_hidden, config.mlp_hidden),
            nn.GELU(),
            nn.Linear(config.mlp_hidden, 1),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        scalars: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        scalar_input = torch.nan_to_num(scalars.float(), nan=0.0, posinf=20.0, neginf=-20.0)
        hidden_features = self.hidden_proj(hidden.float())
        scalar_features = self.scalar_proj(scalar_input)
        logits = self.net(torch.cat([hidden_features, scalar_features], dim=-1)).squeeze(-1)
        mask = torch.sigmoid(logits / self.config.temperature)
        return mask * valid.float()


class FunctionalLoRALinear(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        out_features, in_features = weight.shape
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)
        if bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
        self.lora_A = nn.Parameter(
            torch.empty(
                rank,
                in_features,
                device=weight.device,
                dtype=weight.dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                out_features,
                rank,
                device=weight.device,
                dtype=weight.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.fast_A: torch.Tensor | None = None
        self.fast_B: torch.Tensor | None = None

    @classmethod
    def from_linear(
        cls,
        module: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> "FunctionalLoRALinear":
        return cls(module.weight, module.bias, rank, alpha, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora_A = self.fast_A if self.fast_A is not None else self.lora_A
        lora_B = self.fast_B if self.fast_B is not None else self.lora_B
        low_rank = F.linear(self.dropout(x), lora_A)
        delta = F.linear(low_rank, lora_B) * self.scaling
        return base + delta


class FunctionalLoRAConv1D(nn.Module):
    """LoRA wrapper for transformers.pytorch_utils.Conv1D used by GPT-2."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        in_features, out_features = weight.shape
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)
        self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
        self.lora_A = nn.Parameter(
            torch.empty(
                rank,
                in_features,
                device=weight.device,
                dtype=weight.dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                out_features,
                rank,
                device=weight.device,
                dtype=weight.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.fast_A: torch.Tensor | None = None
        self.fast_B: torch.Tensor | None = None

    @classmethod
    def from_conv1d(cls, module, rank: int, alpha: float, dropout: float):
        return cls(module.weight, module.bias, rank, alpha, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size_out = x.size()[:-1] + (self.out_features,)
        base = torch.addmm(
            self.bias,
            x.reshape(-1, x.size(-1)),
            self.weight,
        ).view(size_out)
        lora_A = self.fast_A if self.fast_A is not None else self.lora_A
        lora_B = self.fast_B if self.fast_B is not None else self.lora_B
        low_rank = F.linear(self.dropout(x), lora_A)
        delta = F.linear(low_rank, lora_B) * self.scaling
        return base + delta


def install_functional_lora(
    model: nn.Module,
    target_modules: list[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> int:
    """Replace matching Linear/Conv1D modules with functional LoRA wrappers."""

    replaced = 0
    try:
        from transformers.pytorch_utils import Conv1D
    except Exception:  # pragma: no cover - transformers always has this in normal runs
        Conv1D = ()  # type: ignore[assignment]

    for name, module in list(model.named_modules()):
        if not name or not _matches_target(name, target_modules):
            continue
        parent, child_name = _resolve_parent(model, name)
        if isinstance(module, nn.Linear):
            setattr(
                parent,
                child_name,
                FunctionalLoRALinear.from_linear(module, rank, alpha, dropout),
            )
            replaced += 1
        elif Conv1D and isinstance(module, Conv1D):
            setattr(
                parent,
                child_name,
                FunctionalLoRAConv1D.from_conv1d(module, rank, alpha, dropout),
            )
            replaced += 1
    if replaced == 0:
        raise ValueError(
            "No LoRA target modules were replaced. "
            f"Targets: {target_modules}"
        )
    return replaced


def _matches_target(module_name: str, target_modules: list[str]) -> bool:
    return any(module_name.endswith(target) for target in target_modules)


def _resolve_parent(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def iter_lora_modules(model: nn.Module) -> Iterator[FunctionalLoRALinear | FunctionalLoRAConv1D]:
    for module in model.modules():
        if isinstance(module, (FunctionalLoRALinear, FunctionalLoRAConv1D)):
            yield module


def iter_lora_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    for module in iter_lora_modules(model):
        yield module.lora_A
        yield module.lora_B


def copy_lora_state(
    model: nn.Module,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    state = {}
    for name, param in model.named_parameters():
        if "lora_" not in name:
            continue
        value = param.detach().clone()
        if device is not None:
            value = value.to(device)
        state[name] = value
    return state


def load_lora_state(
    model: nn.Module,
    state: dict[str, torch.Tensor],
    device: torch.device | None = None,
    strict: bool = True,
) -> None:
    named_params = {
        name: param for name, param in model.named_parameters() if "lora_" in name
    }
    missing = sorted(set(named_params) - set(state))
    unexpected = sorted(set(state) - set(named_params))
    if strict and (missing or unexpected):
        raise ValueError(
            "LoRA state mismatch. "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    with torch.no_grad():
        for name, value in state.items():
            if name not in named_params:
                continue
            param = named_params[name]
            target = value.to(
                device=device or param.device,
                dtype=param.dtype,
            )
            if param.shape != target.shape:
                raise ValueError(
                    f"LoRA parameter shape mismatch for {name}: "
                    f"{tuple(param.shape)} vs {tuple(target.shape)}"
                )
            param.copy_(target)


def make_fast_lora_state(
    model: nn.Module,
    grads: list[torch.Tensor | None],
    lr: float,
) -> tuple[dict[nn.Module, dict[str, torch.Tensor]], torch.Tensor]:
    state: dict[nn.Module, dict[str, torch.Tensor]] = {}
    grad_norm_sq = None
    grad_iter = iter(grads)
    for module in iter_lora_modules(model):
        grad_A = next(grad_iter)
        grad_B = next(grad_iter)
        fast_A = module.lora_A if grad_A is None else module.lora_A - lr * grad_A
        fast_B = module.lora_B if grad_B is None else module.lora_B - lr * grad_B
        state[module] = {"A": fast_A, "B": fast_B}
        for grad in (grad_A, grad_B):
            if grad is None:
                continue
            value = grad.detach().float().pow(2).sum()
            grad_norm_sq = value if grad_norm_sq is None else grad_norm_sq + value
    if grad_norm_sq is None:
        device = next(model.parameters()).device
        grad_norm = torch.zeros((), device=device)
    else:
        grad_norm = grad_norm_sq.sqrt()
    return state, grad_norm


@contextmanager
def use_fast_lora(
    state: dict[nn.Module, dict[str, torch.Tensor]],
) -> Iterator[None]:
    previous: list[tuple[nn.Module, torch.Tensor | None, torch.Tensor | None]] = []
    try:
        for module, values in state.items():
            previous.append((module, module.fast_A, module.fast_B))
            module.fast_A = values["A"]
            module.fast_B = values["B"]
        yield
    finally:
        for module, fast_A, fast_B in previous:
            module.fast_A = fast_A
            module.fast_B = fast_B


def lora_parameter_norm(model: nn.Module) -> torch.Tensor:
    total = None
    for param in iter_lora_parameters(model):
        value = param.detach().float().pow(2).sum()
        total = value if total is None else total + value
    if total is None:
        device = next(model.parameters()).device
        return torch.zeros((), device=device)
    return total.sqrt()
