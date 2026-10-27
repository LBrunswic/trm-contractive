"""Pure-PyTorch port of the `adam-atan2==0.0.3` CUDA optimizer.

Derived from adam-atan2 0.0.3 (https://github.com/imoneoi/adam_atan2), Copyright (c) its authors, licensed
under the Apache License, Version 2.0 (see LICENSES/Apache-2.0.txt). The constructor's argument checks follow
adam_atan2/adam_atan2.py of that package. Modified: the CUDA kernel (csrc/adam_atan2.cu) is re-expressed with
torch._foreach ops, so it runs without the compiled extension (e.g. no nvcc). Per element, in fp32 opmath:

    param *= 1 - lr * weight_decay
    exp_avg    = lerp(exp_avg,    grad,        1 - beta1)
    exp_avg_sq = lerp(exp_avg_sq, grad * grad, 1 - beta2)
    param -= lr / (1 - beta1^t) * atan2(exp_avg, sqrt(exp_avg_sq) / sqrt(1 - beta2^t))

`lr=0` is allowed (pretrain.py constructs the optimizer with lr=0 and sets it each step).
The step counter is kept on the CPU so that no host/device sync is needed.
"""
import math
from typing import Tuple, Union

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT


class AdamATan2(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 1e-2
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = float(group["lr"])
            weight_decay = group["weight_decay"]

            # Bucket by (step, device, dtype) so foreach kernels see homogeneous lists
            buckets = {}
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("AdamATan2 does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.zeros((), dtype=torch.float32)  # CPU, like a host counter
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                if state["step"].device.type != "cpu":  # e.g. after load_state_dict(map_location="cuda")
                    state["step"] = state["step"].cpu()
                state["step"] += 1
                key = (float(state["step"]), p.device, p.dtype)
                bucket = buckets.setdefault(key, ([], [], [], []))
                bucket[0].append(p)
                bucket[1].append(p.grad)
                bucket[2].append(state["exp_avg"])
                bucket[3].append(state["exp_avg_sq"])

            for (step, _device, _dtype), (params, grads, exp_avgs, exp_avg_sqs) in buckets.items():
                _adam_atan2_foreach(params, grads, exp_avgs, exp_avg_sqs,
                                    step=step, lr=lr, beta1=beta1, beta2=beta2, weight_decay=weight_decay)

        return loss


def _adam_atan2_foreach(params, grads, exp_avgs, exp_avg_sqs, *, step: float, lr: float,
                        beta1: float, beta2: float, weight_decay: float) -> None:
    # Scalars computed in double precision, as in the kernel
    step_size = lr / (1 - beta1 ** step)
    wd_alpha = 1 - lr * weight_decay
    bias_correction2_sqrt = math.sqrt(1 - beta2 ** step)

    low_precision = params[0].dtype not in (torch.float32, torch.float64)
    if low_precision:
        # fp32 opmath for half/bf16 parameters
        P, G, M, V = ([t.float() for t in ts] for ts in (params, grads, exp_avgs, exp_avg_sqs))
    else:
        P, G, M, V = params, [g.to(params[0].dtype) for g in grads], exp_avgs, exp_avg_sqs

    torch._foreach_mul_(P, wd_alpha)
    torch._foreach_lerp_(M, G, 1 - beta1)
    torch._foreach_lerp_(V, torch._foreach_mul(G, G), 1 - beta2)

    denom = torch._foreach_sqrt(V)
    torch._foreach_div_(denom, bias_correction2_sqrt)
    torch._foreach_add_(P, [torch.atan2(m, d) for m, d in zip(M, denom)], alpha=-step_size)

    if low_precision:
        for dst, src in zip(params, P):
            dst.copy_(src)
        for dst, src in zip(exp_avgs, M):
            dst.copy_(src)
        for dst, src in zip(exp_avg_sqs, V):
            dst.copy_(src)
