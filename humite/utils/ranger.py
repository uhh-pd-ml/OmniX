"""The Ranger optimizer: RAdam wrapped in Lookahead.

Adapted from weaver-core (https://github.com/hqucms/weaver-core, weaver/utils/nn/optimizer).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from torch.optim import Optimizer


class RAdam(Optimizer):
    """Rectified Adam (https://arxiv.org/abs/1908.03265) with decoupled weight decay."""

    def __init__(
        self, params, lr: float, betas: Tuple[float, float], eps: float, weight_decay: float
    ):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @staticmethod
    def rectification(step: int, beta1: float, beta2: float) -> Tuple[float, float]:
        """Return (length of the approximated simple moving average, step size) at `step`."""
        beta2_t = beta2**step
        sma_max = 2 / (1 - beta2) - 1
        sma = sma_max - 2 * step * beta2_t / (1 - beta2_t)
        if sma < 5:
            return sma, 1.0 / (1 - beta1**step)
        variance = (
            (1 - beta2_t) * (sma - 4) / (sma_max - 4) * (sma - 2) / sma * sma_max / (sma_max - 2)
        )
        return sma, math.sqrt(variance) / (1 - beta1**step)

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            by_step: Dict[int, List[torch.Tensor]] = {}
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                state["step"] += 1
                by_step.setdefault(state["step"], []).append(p)
            for step, params in by_step.items():
                data = [p.data for p in params]
                grads = [p.grad.data for p in params]
                exp_avgs = [self.state[p]["exp_avg"] for p in params]
                exp_avg_sqs = [self.state[p]["exp_avg_sq"] for p in params]
                torch._foreach_mul_(exp_avg_sqs, beta2)
                torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - beta2)
                torch._foreach_mul_(exp_avgs, beta1)
                torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)
                sma, step_size = self.rectification(step, beta1, beta2)
                if group["weight_decay"] != 0:
                    torch._foreach_add_(data, data, alpha=-group["weight_decay"] * group["lr"])
                if sma >= 5:
                    denominator = torch._foreach_sqrt(exp_avg_sqs)
                    torch._foreach_add_(denominator, group["eps"])
                    torch._foreach_addcdiv_(
                        data, exp_avgs, denominator, value=-step_size * group["lr"]
                    )
                else:
                    torch._foreach_add_(data, exp_avgs, alpha=-step_size * group["lr"])
        return loss


class Lookahead(Optimizer):
    """Lookahead (https://arxiv.org/abs/1907.08610): every `k` steps the weights are pulled
    back towards a slow copy, `slow + alpha * (fast - slow)`. The slow copy is not part of
    the saved state; after a restart it starts from the loaded weights."""

    def __init__(self, optimizer: Optimizer, alpha: float, k: int) -> None:
        self.optimizer = optimizer
        self.alpha = alpha
        self.k = k
        self.step_counter = 0
        self.defaults = optimizer.defaults
        self.reset()

    def reset(self) -> None:
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["slow"] = p.data.clone()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict) -> None:
        self.optimizer.load_state_dict(state_dict)
        self.reset()

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self.step_counter += 1
        if self.step_counter >= self.k:
            self.step_counter = 0
            for group in self.param_groups:
                for p in group["params"]:
                    slow = self.state[p]["slow"]
                    p.data.mul_(self.alpha).add_(slow, alpha=1.0 - self.alpha)
                    slow.copy_(p.data)
        return loss


def Ranger(params, lr, betas, eps, weight_decay, lookahead_alpha, lookahead_k) -> Lookahead:
    radam = RAdam(params, lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay)
    return Lookahead(radam, lookahead_alpha, lookahead_k)
