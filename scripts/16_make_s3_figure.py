"""Figure 4 (S3) -- cross-condition transfer under a leakage-free protocol.

Two panels, one per finding:

* **(a) The median-ratio correction annihilates the baseline ladder.** The
  correction is applied in the cross-condition path only and divides the answer
  by a factor fitted on the labelled source split. For a constant predictor that
  factor is exactly ``1 / c``, so *every* constant is mapped onto the same
  prediction, ``median(val_ratio) * lifetime``. Per task we therefore plot three
  quantities: the naive constant (ratio 1.0, uncorrected), the strongest constant
  (mean ratio, uncorrected), and the single value the whole constant family
  collapses to once corrected. Marker at left = uncorrected, black = corrected.

* **(b) No feature-based model clears the hindsight constant floor.** For each
  entry, the mean over the nine cross tasks of ``RMSE / oracle-constant RMSE``
  (log axis). The dashed line at 1.0 is the floor: everything to its right is
  beaten by a constant that was handed the target labels.

Inputs are the artefacts written by ``scripts/15_cross_transfer.py`` and the two
run directories, so the numbers cannot drift from the tables.

    python scripts/16_make_s3_figure.py
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
})

C_ONE = "#999999"
C_MEAN = "#c0504d"
C_CORR = "#000000"
C_MODEL = "#1f4e79"
C_DEEP = "#2e8b57"

SUMMARY = REPO / "artifacts" / "reports" / "cross_transfer_summary.json"
UDA = REPO / "artifacts" / "reports" / "cross_uda_linear"
SRC = REPO / "artifacts" / "reports" / "cross_source_only_linear"

ORDER = ["oracle const (task)", "corrected const", "const (mean) uncorr",
         "const (one) uncorr", "ridge", "cnn", "lstm", "gru", "SE-ResNet-BiGRU"]


def read_csv(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def pivot(rows: list[dict], metric: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for r in rows:
        try:
            out.setdefault(r["dataset"], {})[r["model"]] = float(r[metric])
        except (KeyError, ValueError):
            continue
    return out


def main() -> int:
    if not SUMMARY.exists():
        print(f"[error] {SUMMARY.relative_to(REPO)} missing -- run scripts/15_cross_transfer.py first")
        return 2
    summ = json.loads(SUMMARY.read_text(encoding="utf-8"))
    anchors = summ["anchors"]
    tasks = sorted(anchors)

    corr = pivot(read_csv(UDA / "results_corrected.csv"), "nrmse")
    unc = pivot(read_csv(UDA / "results_uncorrected.csv"), "nrmse")
    corr_rmse = pivot(read_csv(UDA / "results_corrected.csv"), "rmse")
    unc_rmse = pivot(read_csv(UDA / "results_uncorrected.csv"), "rmse")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.7),
                                   gridspec_kw={"width_ratios": [1.55, 1.0]})

    # ---------------------------------------------------------------- panel (a)
    x = np.arange(len(tasks))
    w = 0.38
    one_u = [unc.get(t, {}).get("constant_one", np.nan) for t in tasks]
    mean_u = [unc.get(t, {}).get("constant_mean", np.nan) for t in tasks]
    corr_v = [corr.get(t, {}).get("constant_one", np.nan) for t in tasks]

    ax1.bar(x - w / 2, one_u, w, color=C_ONE, edgecolor="white", linewidth=0.4,
            label="const (one), uncorrected")
    ax1.bar(x + w / 2, mean_u, w, color=C_MEAN, edgecolor="white", linewidth=0.4,
            label="const (mean), uncorrected")
    ax1.plot(x, corr_v, "D", color=C_CORR, markersize=3.6, zorder=5,
             label="corrected: any constant collapses here")

    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks, rotation=35, ha="right")
    ax1.set_ylabel("NRMSE (%)")
    ax1.set_title("(a) the ratio correction collapses the baseline ladder")
    ax1.legend(loc="upper left", frameon=False, ncol=1)

    # ---------------------------------------------------------------- panel (b)
    def _ratio_mean(metric: dict[str, dict[str, float]], model: str) -> float:
        vals = []
        for t in tasks:
            v = metric.get(t, {}).get(model)
            base = anchors[t]["oracle_const_task__rmse"]
            if v is not None and base:
                vals.append(v / base)
        return float(np.mean(vals)) if vals else np.nan

    entries = [
        ("oracle const (task)", 1.0, C_CORR, 0.0),
        ("corrected const", _ref_corr(corr_rmse, anchors, tasks), C_CORR, 0.0),
        ("const (mean) uncorr", _ratio_mean(unc_rmse, "constant_mean"), C_MEAN, 0.0),
        ("const (one) uncorr", _ratio_mean(unc_rmse, "constant_one"), C_ONE, 0.0),
        ("ridge", _ratio_mean(corr_rmse, "ridge"), C_ONE, 0.0),
        ("cnn", _ratio_mean(corr_rmse, "cnn"), C_DEEP, 0.0),
        ("lstm", _ratio_mean(corr_rmse, "lstm"), C_DEEP, 0.0),
        ("gru", _ratio_mean(corr_rmse, "gru"), C_DEEP, 0.0),
        ("SE-ResNet-BiGRU", _ratio_mean(corr_rmse, "se_resnet_bigru"), C_MODEL, 0.0),
    ]
    ypos = np.arange(len(entries))[::-1]
    for (name, val, color, _), y in zip(entries, ypos):
        if not np.isfinite(val):
            continue
        ax2.barh(y, val, height=0.62, color=color, edgecolor="white", linewidth=0.4)
        ax2.text(val * 1.06, y, f"{val:.2f}", va="center", fontsize=7)

    ax2.axvline(1.0, color=C_CORR, linestyle="--", linewidth=0.9)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([n for n, _, _, _ in entries])
    # every value is >= 1, so a linear axis reads better than a log one and still
    # separates the deployable constant (1.04) from the hindsight floor (1.00)
    ax2.set_xlim(0.85, 2.62)
    ax2.set_xticks([1.0, 1.5, 2.0, 2.5])
    ax2.set_xlabel("mean RMSE / oracle-constant RMSE")
    ax2.set_title("(b) nothing clears the hindsight constant floor")
    ax2.grid(axis="y", visible=False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig4_cross_transfer.{ext}")
    plt.close(fig)
    print(f"wrote {OUT.relative_to(REPO)}/fig4_cross_transfer.[png|pdf]")
    print("(b) ratios:", {n: round(v, 3) for n, v, _, _ in entries if np.isfinite(v)})
    return 0


def _ref_corr(corr_rmse, anchors, tasks) -> float:
    """Ratio of the corrected constant to the oracle, per task, then averaged."""
    vals = []
    for t in tasks:
        v = corr_rmse.get(t, {}).get("constant_one")
        base = anchors[t]["oracle_const_task__rmse"]
        if v is not None and base:
            vals.append(v / base)
    return float(np.mean(vals)) if vals else float("nan")


if __name__ == "__main__":
    sys.exit(main())
