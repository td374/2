"""One-off probe: windowed trivial-baseline anchors under piecewise vs linear labels.

Not part of the package. Deleted after use.
"""
from __future__ import annotations

import numpy as np

LIVES = {"1_5": (52, 52), "2_5": (339, 339), "3_5": (114, 114)}  # n_samples, lifetime_min
WINDOW = 32
H = 0.8


def piecewise(n, h=H):
    thr = max(1, min(int(n * h), n - 1))
    span = n - thr
    r = np.ones(n)
    r[thr:] = (n - np.arange(thr, n)) / span
    return r


def linear(n):
    r = np.ones(n)
    r[:] = (np.arange(1, n + 1) / n)[::-1]
    return r


def anchors(n, life):
    ends = np.arange(WINDOW - 1, n)
    out = {}
    for tag, ratio in (("piecewise", piecewise(n)), ("linear", linear(n))):
        r = ratio[ends]
        c1 = 1.0
        cm = float(r.mean())
        out[tag] = {
            "const_one": life * float(np.sqrt(np.mean((r - c1) ** 2))),
            "const_mean": life * float(np.sqrt(np.mean((r - cm) ** 2))),
            "const_zero": life * float(np.sqrt(np.mean(r**2))),
            "mean_ratio": cm,
            "opt_const": life * float(r.std()),  # best constant = mean
        }
    return out


print(f"window={WINDOW}  h={H}")
print(f"{'bearing':>8} {'n':>5} {'label':>10} {'const_one':>10} {'const_mean':>11} {'const_zero':>11} {'mean_ratio':>11}")
for name, (n, life) in LIVES.items():
    a = anchors(n, life)
    for tag in ("piecewise", "linear"):
        d = a[tag]
        print(f"{name:>8} {n:>5} {tag:>10} {d['const_one']:>10.2f} {d['const_mean']:>11.2f} {d['const_zero']:>11.2f} {d['mean_ratio']:>11.4f}")
    print()

# 3-condition means
for tag in ("piecewise", "linear"):
    tot = {k: np.mean([anchors(n, life)[tag][k] for n, life in LIVES.values()]) for k in ("const_one", "const_mean", "const_zero")}
    print(f"mean over 3 test bearings [{tag}]: const_one={tot['const_one']:.2f}  const_mean={tot['const_mean']:.2f}  const_zero={tot['const_zero']:.2f}")
