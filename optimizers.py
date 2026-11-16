"""Optimizers by name (train.optimizer), with their extra arguments from train.optimizer_args.

The benchmark uses AdamW; the others exist for the optimizer ablation. Weight decay is
train.weight_decay for all of them: decoupled for AdamW and Lion, an L2 term in the
gradient for the rest (their standard definitions).
"""

from __future__ import annotations

from typing import Any

import torch

TORCH_OPTIMIZERS = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "nadam": torch.optim.NAdam,
    "radam": torch.optim.RAdam,
    "adamax": torch.optim.Adamax,
    "rmsprop": torch.optim.RMSprop,
    "adagrad": torch.optim.Adagrad,
}


class Lion(torch.optim.Optimizer):
    """Lion (Chen et al., "Symbolic Discovery of Optimization Algorithms", NeurIPS 2023).

    The update is the sign of an interpolation of the momentum and the gradient, so every
    weight moves by exactly lr per step: use a 3-10x smaller lr and a 3-10x larger weight
    decay than for AdamW. Weight decay is decoupled, as in AdamW.
    """

    def __init__(self, params, lr: float = 1e-4, betas: tuple[float, float] = (0.9, 0.99), weight_decay: float = 0.0):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params, {"lr": lr, "betas": tuple(betas), "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if not state:
                    state["exp_avg"] = torch.zeros_like(parameter)
                momentum = state["exp_avg"]
                if group["weight_decay"]:
                    parameter.mul_(1 - group["lr"] * group["weight_decay"])
                update = momentum.mul(beta1).add_(parameter.grad, alpha=1 - beta1).sign_()
                parameter.add_(update, alpha=-group["lr"])
                momentum.mul_(beta2).add_(parameter.grad, alpha=1 - beta2)
        return loss


OPTIMIZERS = (*TORCH_OPTIMIZERS, "lion")


def build_optimizer(groups: list[dict[str, Any]], train: dict[str, Any]) -> torch.optim.Optimizer:
    """Optimizer `train.optimizer` over parameter `groups` (each with its own "lr")."""
    name = str(train.get("optimizer", "adamw")).lower()
    options = dict(train.get("optimizer_args") or {})
    if "betas" in options:
        options["betas"] = tuple(options["betas"])
    if name == "lion":
        return Lion(groups, lr=train["lr"], weight_decay=train["weight_decay"], **options)
    if name not in TORCH_OPTIMIZERS:
        raise ValueError(f"Unknown optimizer '{name}'. Choose one of: {', '.join(OPTIMIZERS)}")
    return TORCH_OPTIMIZERS[name](groups, lr=train["lr"], weight_decay=train["weight_decay"], **options)
