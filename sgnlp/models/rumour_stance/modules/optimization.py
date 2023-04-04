import math
from typing import Callable

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.optimizer import required


def warmup_cosine(x, warmup=0.002):
    """Calculate warmup_cosine learning rate scheduler."""
    if x < warmup:
        return x / warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))


def warmup_constant(x, warmup=0.002):
    """Calculate warmup_constant learning rate scheduler."""
    if x < warmup:
        return x / warmup
    return 1.0


def warmup_linear(x, warmup=0.002):
    """Calculate warmup_linear learning rate scheduler."""
    if x < warmup:
        return x / warmup
    return 1.0 - x


SCHEDULES = {
    "warmup_cosine": warmup_cosine,
    "warmup_constant": warmup_constant,
    "warmup_linear": warmup_linear,
}


class BertAdam(Optimizer):
    """Implement BERT version of Adam algorithm with weight decay fix.

    Args:
        params: Parameters.
        lr (float, optional): Learning rate. Defaults to required.
        warmup (float, optional): Portion of t_total for the warmup, -1  means no warmup. Defaults to -1.
        t_total (int, optional): Total number of training steps for the learning rate schedule, -1  means constant learning rate. Defaults to -1.
        schedule (str, optional): Schedule to use for the warmup. Defaults to "warmup_linear".
        b1 (float, optional): Adams b1. Defaults to 0.9.
        b2 (float, optional): Adams b2. Defaults to 0.999.
        e (float, optional): Adams epsilon. Defaults to 1e-6.
        weight_decay (float, optional): Weight decay. Defaults to 0.01.
        max_grad_norm (float, optional): Maximum norm for the gradients (-1 means no clipping). Defaults to 1.0.
    """

    def __init__(
        self,
        params,
        lr: float = required,
        warmup: float = -1,
        t_total: int = -1,
        schedule: str = "warmup_linear",
        b1: float = 0.9,
        b2: float = 0.999,
        e: float = 1e-6,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
    ) -> None:
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError(
                "Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup)
            )
        if not 0.0 <= b1 < 1.0:
            raise ValueError(
                "Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1)
            )
        if not 0.0 <= b2 < 1.0:
            raise ValueError(
                "Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2)
            )
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(
            lr=lr,
            schedule=schedule,
            warmup=warmup,
            t_total=t_total,
            b1=b1,
            b2=b2,
            e=e,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
        )
        super(BertAdam, self).__init__(params, defaults)

    def get_lr(self) -> list:
        """Calculate the learning rate for each training step"""
        lr = []
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group["t_total"] != -1:
                    schedule_fct = SCHEDULES[group["schedule"]]
                    lr_scheduled = group["lr"] * schedule_fct(
                        state["step"] / group["t_total"], group["warmup"]
                    )
                else:
                    lr_scheduled = group["lr"]
                lr.append(lr_scheduled)
        return lr

    def step(self, closure: Callable = None):
        """Perform a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
            float: Loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["next_m"] = torch.zeros_like(p.data)
                    state["next_v"] = torch.zeros_like(p.data)

                next_m, next_v = state["next_m"], state["next_v"]
                beta1, beta2 = group["b1"], group["b2"]

                if group["max_grad_norm"] > 0:
                    clip_grad_norm_(p, group["max_grad_norm"])

                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group["e"])

                if group["weight_decay"] > 0.0:
                    update += group["weight_decay"] * p.data

                if group["t_total"] != -1:
                    schedule_fct = SCHEDULES[group["schedule"]]
                    lr_scheduled = group["lr"] * schedule_fct(
                        state["step"] / group["t_total"], group["warmup"]
                    )
                else:
                    lr_scheduled = group["lr"]

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state["step"] += 1
        return loss
