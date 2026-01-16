# src/lora_utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Any

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with LoRA adapters:

        y = xW^T + b + (alpha/r) * B(A(dropout(x)))

    Base weights are frozen; only LoRA A/B are trained.
    """
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")

        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features

        # Freeze base layer
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = self.alpha / self.r

        self.lora_dropout = nn.Dropout(p=float(dropout)) if dropout and dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(self.in_features, self.r, bias=False)
        self.lora_B = nn.Linear(self.r, self.out_features, bias=False)

        # LoRA init: A random, B zeros (so initial LoRA contribution is 0)
        nn.init.normal_(self.lora_A.weight, std=0.01)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(self.lora_dropout(x)))


def _get_module(model: nn.Module, name: str) -> nn.Module:
    cur = model
    for part in name.split("."):
        cur = getattr(cur, part)
    return cur


def _set_module(model: nn.Module, name: str, module: nn.Module) -> None:
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], module)


def find_linear_module_names(
    model: nn.Module,
    include_substring: str = "text_decoder",
    suffixes: List[str] | Tuple[str, ...] = ("query", "key", "value"),
) -> List[str]:
    """
    Find nn.Linear modules whose full name contains include_substring
    and whose last component is in suffixes.
    """
    names: List[str] = []
    suffix_set = set(suffixes)

    for name, mod in model.named_modules():
        if include_substring not in name:
            continue
        if not isinstance(mod, nn.Linear):
            continue
        last = name.split(".")[-1]
        if last in suffix_set:
            names.append(name)

    return sorted(names)


def inject_lora(
    model: nn.Module,
    module_names: List[str],
    r: int,
    alpha: int,
    dropout: float,
) -> int:
    """
    Replace the specified nn.Linear modules with LoRALinear wrappers.
    Returns number of modules replaced.
    """
    replaced = 0
    for name in module_names:
        mod = _get_module(model, name)
        if not isinstance(mod, nn.Linear):
            continue
        _set_module(model, name, LoRALinear(mod, r=r, alpha=alpha, dropout=dropout))
        replaced += 1
    return replaced


def mark_only_lora_trainable(model: nn.Module) -> int:
    """
    Freeze everything except LoRA A/B weights.
    Returns number of trainable parameters.
    """
    for p in model.parameters():
        p.requires_grad = False

    for n, p in model.named_parameters():
        if ".lora_A." in n or ".lora_B." in n:
            p.requires_grad = True

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def extract_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Save only LoRA weights.
    """
    sd = model.state_dict()
    keep = {k: v.detach().cpu() for k, v in sd.items() if ".lora_A." in k or ".lora_B." in k}
    return keep


def load_lora_state_dict(model: nn.Module, lora_sd: Dict[str, torch.Tensor]) -> None:
    """
    Load only LoRA weights into a model that already has LoRA injected.
    """
    missing, unexpected = model.load_state_dict(lora_sd, strict=False)
    # These are expected because we intentionally load only LoRA weights.
    # But if LoRA wasn't injected correctly, you'll see a lot of missing keys.
    if len(unexpected) > 0:
        print("[LoRA] Unexpected keys:", unexpected[:10])
    if len(missing) > 0:
        # Print only a few to avoid spam
        print("[LoRA] Missing keys (sample):", missing[:10])
