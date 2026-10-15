"""Loss functions: weighted MSE and an efficient multi-kernel MMD.

Why the MMD implementation differs from the thesis version
----------------------------------------------------------
The original code expanded absolute feature differences into a
``(n, n, 4096)`` tensor before summing. With a batch of 128 source plus 128
target windows that materialises ~268 M floats (about 1 GiB) per kernel, five
times over, and keeps the GPU busy on pure memory traffic.

Computing the squared distances through the Gram trick,
``||x||^2 + ||y||^2 - 2 x y^T``, reduces the intermediate to a ``(n, n)``
matrix -- the exact same quantity with roughly three orders of magnitude less
memory and a large speed-up, which matters because this is the dominant cost of
the cross-condition runs.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn as nn

# --------------------------------------------------------------------------- MSE
def weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.3,
    weight: float = 2.0,
) -> torch.Tensor:
    """MSE that upweights the accelerating-degradation phase.

    ``target`` is the normalised RUL ratio, so ``target < threshold`` marks the
    last ``threshold`` fraction of life -- the region where an under-estimate is
    safety-critical.
    """
    w = torch.where(
        target < threshold,
        torch.full_like(target, float(weight)),
        torch.ones_like(target),
    )
    return torch.mean(w * (pred - target) ** 2)


def plain_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(pred, target)


LOSS_FACTORIES: dict[str, Callable[..., torch.Tensor]] = {
    "weighted_mse": weighted_mse,
    "mse": plain_mse,
}


def build_supervised_loss(name: str, **kwargs) -> Callable[..., torch.Tensor]:
    if name not in LOSS_FACTORIES:
        raise KeyError(f"unknown supervised loss {name!r}; options: {sorted(LOSS_FACTORIES)}")
    fn = LOSS_FACTORIES[name]

    def wrapped(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return fn(pred, target, **kwargs)

    return wrapped


# ------------------------------------------------------------------------ MMD
def pairwise_sq_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distances via the Gram trick, returning ``(nx, ny)``."""
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True)
    dist = x2 + y2.transpose(0, 1) - 2.0 * (x @ y.transpose(0, 1))
    return dist.clamp_min_(0.0)


def multi_kernel_mmd(
    source: torch.Tensor,
    target: torch.Tensor,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: float | None = None,
    unbiased: bool = False,
) -> torch.Tensor:
    """MK-MMD between two batches of representations.

    Parameters
    ----------
    kernel_mul, kernel_num
        Gaussian bandwidths are ``bw * kernel_mul**i`` for
        ``i in [0, kernel_num)``, with ``bw`` from the median heuristic over the
        pooled batch. Five kernels is the DAN default and is what the thesis
        used.
    unbiased
        ``False`` reproduces the original estimator (diagonal included), which
        stays non-negative and is the common choice for a training penalty.
    """
    n_s = source.size(0)
    n_t = target.size(0)
    if n_s == 0 or n_t == 0:
        return source.new_zeros(())

    pooled = torch.cat([source, target], dim=0)
    dist = pairwise_sq_dist(pooled, pooled)                       # (n, n)
    n = pooled.size(0)

    if fix_sigma is not None:
        bandwidth = torch.as_tensor(fix_sigma, device=pooled.device, dtype=pooled.dtype)
    else:
        # median heuristic; exclude the zero diagonal
        bandwidth = dist.detach().sum() / max(n * (n - 1), 1)
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))

    kernel = torch.zeros_like(dist)
    for i in range(kernel_num):
        kernel = kernel + torch.exp(-dist / (bandwidth * (kernel_mul ** i)))

    xx = kernel[:n_s, :n_s]
    yy = kernel[n_s:, n_s:]
    xy = kernel[:n_s, n_s:]
    if unbiased:
        xx = xx[~torch.eye(n_s, dtype=torch.bool, device=xx.device)].view(n_s, n_s - 1)
        yy = yy[~torch.eye(n_t, dtype=torch.bool, device=yy.device)].view(n_t, n_t - 1)
        # symmetric XY: average the two off-diagonal blocks
        xy_sym = 0.5 * (xy + kernel[n_s:, :n_s])
        return xx.mean() + yy.mean() - 2.0 * xy_sym.mean()
    return xx.mean() + yy.mean() - xy.mean() - kernel[n_s:, :n_s].mean()


#: backwards-compatible alias matching the legacy function name
mk_mmd = multi_kernel_mmd


class CombinedLoss(nn.Module):
    """``supervised + lambda * mmd`` with the components exposed for logging."""

    def __init__(
        self,
        supervised: Callable[..., torch.Tensor],
        lambda_mmd: float = 0.1,
        lambda_schedule: Sequence[float] | None = None,
    ):
        super().__init__()
        self.supervised = supervised
        self.lambda_mmd = float(lambda_mmd)
        self.lambda_schedule = list(lambda_schedule) if lambda_schedule else None

    def current_lambda(self, epoch: int) -> float:
        """Optionally ramp lambda (e.g. ``[0.0, 0.01, 0.1]`` over epochs)."""
        if not self.lambda_schedule:
            return self.lambda_mmd
        idx = min(epoch, len(self.lambda_schedule) - 1)
        return float(self.lambda_schedule[idx])

    def forward(
        self,
        pred_src: torch.Tensor,
        y_src: torch.Tensor,
        feat_src: torch.Tensor | None = None,
        feat_tgt: torch.Tensor | None = None,
        epoch: int = 0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        task = self.supervised(pred_src, y_src)
        lam = self.current_lambda(epoch)
        if feat_src is None or feat_tgt is None or lam == 0.0:
            return task, {"task": float(task.detach()), "mmd": 0.0, "lambda": lam}
        mmd = multi_kernel_mmd(feat_src, feat_tgt)
        return task + lam * mmd, {"task": float(task.detach()), "mmd": float(mmd.detach()), "lambda": lam}
