"""Stage 2.1 figures: prediction-vs-truth scatter and the bias profile.

Consumes ``artifacts/reports/stage2_bias_scatter.json`` (written by
``scripts/26_stage2_bias.py``) and produces

* ``fig_stage21_pred_vs_true.png``  -- 2x3 small multiples, predicted vs true
  **normalised** RUL, one panel per label point, coloured by test bearing;
* ``fig_stage21_bias_profile.png``  -- mean prediction error in normalised RUL
  units against the true normalised RUL, one curve per label point.

Both use normalised RUL on purpose: the three test bearings live 52 / 339 / 114
minutes, so a shared minutes axis collapses two of them into a vertical stripe
and makes the mechanism unreadable.

    python scripts/27_stage21_figure.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]

ORDER = ["linear", "h=0.5", "h=0.6", "h=0.7", "h=0.8", "h=0.9"]
COLORS = ["#1f4e79", "#2e75b6", "#7f7f7f", "#c55a11", "#c00000", "#7030a0"]
BEARING_COLORS = ["#1f4e79", "#c55a11", "#2e8b57"]


def _load() -> dict:
    path = REPO / "artifacts" / "reports" / "stage2_bias_scatter.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for label in ORDER:
        holders = data.get(label, {})
        keys = sorted(holders)
        ratio = np.concatenate([np.asarray(holders[k]["ratio"], float) for k in keys])
        y_true = np.concatenate([np.asarray(holders[k]["y_true"], float) for k in keys])
        y_pred = np.concatenate([np.asarray(holders[k]["y_pred"], float) for k in keys])
        # lifetime per window is recoverable from y_true / ratio (both stored)
        safe = np.where(np.abs(ratio) > 1e-9, ratio, np.nan)
        life = np.concatenate([
            np.full(len(holders[k]["ratio"]),
                    float(np.nanmean(np.asarray(holders[k]["y_true"], float)
                                     / np.where(np.abs(np.asarray(holders[k]["ratio"], float)) > 1e-9,
                                                np.asarray(holders[k]["ratio"], float), np.nan))))
            for k in keys])
        out[label] = {
            "ratio": ratio,
            "pred_ratio": y_pred / life,
            "grp": np.concatenate([np.full(len(holders[k]["ratio"]), gi) for gi, k in enumerate(keys)]),
        }
    return out


def fig_scatter(data: dict) -> pathlib.Path:
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.0), sharex=True, sharey=True)
    for ax, label in zip(axes.ravel(), ORDER):
        dd = data[label]
        for g in range(int(dd["grp"].max()) + 1):
            m = dd["grp"] == g
            ax.scatter(dd["ratio"][m], dd["pred_ratio"][m], s=8, alpha=0.45,
                       color=BEARING_COLORS[g % len(BEARING_COLORS)], edgecolors="none")
        ax.plot([0, 1], [0, 1], color="#888888", lw=1.0, ls="--")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{label}   (n={dd['ratio'].size})", fontsize=10)
        ax.grid(alpha=0.25, lw=0.6)
        ax.tick_params(labelsize=8)
    for ax in axes[1]:
        ax.set_xlabel("true RUL fraction  (1 = new)", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("predicted RUL fraction", fontsize=9)
    axes[0, 0].legend([f"bearing {i + 1}_5" for i in range(3)], fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Stage 2.1 — predicted vs true RUL fraction (se_resnet_bigru); "
                 "points below the diagonal are RUL under-estimates", fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = REPO / "paper" / "figures" / "fig_stage21_pred_vs_true.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_profile(data: dict) -> pathlib.Path:
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    edges = np.linspace(0.0, 1.0, 6)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.axhline(0.0, color="#888888", lw=1.0, ls=":")
    for label, color in zip(ORDER, COLORS):
        dd = data[label]
        ratio, pred_ratio = dd["ratio"], dd["pred_ratio"]
        idx = np.clip(np.digitize(ratio, edges) - 1, 0, len(centers) - 1)
        means = [np.nanmean((pred_ratio - ratio)[idx == k]) if np.any(idx == k) else np.nan
                 for k in range(len(centers))]
        ax.plot(centers, means, "o-", color=color, lw=1.8, ms=5, label=label)
    ax.set_xlabel("true normalised RUL  (1 = new, 0 = failure)", fontsize=10)
    ax.set_ylabel(r"mean $(\hat r - r)$   [RUL fraction]", fontsize=10)
    ax.set_title("Stage 2.1 — bias profile: where the error comes from", fontsize=11)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=8, ncol=2, frameon=False)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    out = REPO / "paper" / "figures" / "fig_stage21_bias_profile.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> int:
    data = _load()
    for p in (fig_scatter(data), fig_profile(data)):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
