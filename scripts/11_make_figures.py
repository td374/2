"""Publication figures for the label-sensitivity study.

Produces, under ``paper/figures/`` (PNG + PDF, English labels, Arial):

* ``fig1_label_design``     -- label shapes (linear vs piecewise plateaus) and
                               how the analytic label informativeness LI falls
                               as the plateau grows.
* ``fig2_spectrum``         -- **headline**: model vs trivial-baseline RMSE,
                               and LI, across the label spectrum; the crossover
                               plateau length where models stop winning.
* ``fig3_main_results``     -- per-condition main results on XJTU-SY and IEEE
                               PHM 2012 (proposed vs baselines, against the
                               analytic best-constant hurdle).

All anchors are read from the *stored* test labels in the prepared artefacts,
so the numbers here are exactly the ones the models were scored against.

    python scripts/11_make_figures.py
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
sys.path.insert(0, str(REPO))

from rulbench.data.labels import linear_rul, piecewise_linear_rul  # noqa: E402

OUT = REPO / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

#: Stage-5 post-fix evidence root (audit/STAGE5_PROTOCOL_CHECK.md)
S5 = REPO / "artifacts" / "reports" / "s5"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
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

C_PROP = "#1f4e79"    # proposed model
C_GRU = "#2e8b57"     # second deep model
C_BIGRU = "#8c6bb1"   # plain BiGRU (no SE / no residual extractor)
C_CMEAN = "#c0504d"   # strongest trivial baseline
C_ONE = "#999999"     # naive baseline
C_HURDLE = "#000000"  # best-constant hurdle
C_LI = "#7f7f7f"

MODEL_LABEL = {
    "se_resnet_bigru": "SE-ResNet-BiGRU (proposed)",
    "gru": "GRU",
    "pure_bigru": "BiGRU",
    "resnet_bigru": "ResNet-BiGRU",
    "se_resnet_gru": "SE-ResNet-GRU",
    "constant_mean": "Constant (mean)",
    "constant_one": "Constant (one)",
}


def read_csv(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def cond_rmse(report_root: pathlib.Path, model: str) -> dict:
    rows = read_csv(report_root / model / "results_corrected.csv")
    return {r["dataset"]: float(r["rmse"]) for r in rows}


def analytic_anchors(data_dir: pathlib.Path) -> dict:
    """{dataset: {const_one, best_const}} from the stored test labels."""
    index = json.loads((data_dir / "index.json").read_text(encoding="utf-8"))
    out: dict = {}
    for rec in index:
        if rec.get("protocol") != "single":
            continue
        with np.load(rec["file"]) as d:
            if "test/ratio" not in d.files:
                continue
            ratio = np.asarray(d["test/ratio"], dtype=np.float64)
            life = np.asarray(d["test/lifetime"], dtype=np.float64)
        # Root-mean-square, in minutes. NB: the previous version multiplied by
        # ``life`` and then took the MEAN of ``|ratio-1|`` -- i.e. it reported MAE
        # while labelling it RMSE (hurdle understated by ~13%). See audit D1.
        one = float(np.sqrt(np.mean((life * (ratio - 1.0)) ** 2)))
        best_c = float(np.sqrt(np.mean((life * (ratio - ratio.mean())) ** 2)))
        out[rec["name"]] = {"const_one": one, "best_const": best_c}
    return out


# --------------------------------------------------------------------- fig 1
def fig1() -> None:
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))

    life = np.linspace(0, 1, 400)
    n = 400
    ax[0].plot(life, linear_rul(n), color=C_PROP, lw=1.8, label="linear (proposed)")
    for h, ls in ((0.5, "--"), (0.8, "-."), (0.9, ":")):
        ax[0].plot(life, piecewise_linear_rul(n, healthy_ratio=h), color="#888888", ls=ls,
                   lw=1.3, label=f"piecewise, h={h}")
    ax[0].set_xlabel("normalised operating life")
    ax[0].set_ylabel("RUL ratio (label)")
    ax[0].set_title("(a) Label definitions")
    ax[0].legend(frameon=False, loc="lower left")
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1.05)

    spec = json.loads((S5 / "label_sensitivity.json").read_text(encoding="utf-8"))
    xs, li = [], []
    for row in spec["points"]:
        lab = row["label"]
        xs.append(0.0 if lab == "linear" else float(lab.split("=")[1]))
        li.append(row["anchors"]["LI"])
    order = np.argsort(xs)
    xs = np.array(xs)[order]
    li = np.array(li)[order]
    ax[1].plot(xs, li, "-o", color=C_LI, ms=4, lw=1.5)
    ax[1].set_xlabel("plateau length $h$  (0 = linear)")
    ax[1].set_ylabel("label informativeness LI")
    ax[1].set_title("(b) Benchmark discriminative power")
    ax[1].set_ylim(0, 1.0)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig1_label_design.{ext}")
    plt.close(fig)


# --------------------------------------------------------------------- fig 2
def fig2() -> None:
    spec = json.loads((S5 / "label_sensitivity.json").read_text(encoding="utf-8"))
    xs, labels = [], []
    li, best, cone = [], [], []
    prop, gru, cmean = [], [], []
    for row in spec["points"]:
        lab = row["label"]
        xs.append(0.0 if lab == "linear" else float(lab.split("=")[1]))
        labels.append("linear" if lab == "linear" else lab)
        li.append(row["anchors"]["LI"])
        best.append(row["anchors"]["best_const"])
        cone.append(row["anchors"]["const_one"])
        prop.append(row["models"].get("se_resnet_bigru", {}).get("mean", np.nan))
        gru.append(row["models"].get("gru", {}).get("mean", np.nan))
        cmean.append(row["models"].get("constant_mean", {}).get("mean", np.nan))
    o = np.argsort(xs)
    xs = np.array(xs)[o]
    li, best, cone = np.array(li)[o], np.array(best)[o], np.array(cone)[o]
    prop, gru, cmean = np.array(prop)[o], np.array(gru)[o], np.array(cmean)[o]
    labels = [labels[i] for i in o]

    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)

    ax.plot(xs, cone, ":", color=C_ONE, lw=1.4, label="Constant (one) — naive")
    ax.plot(xs, best, "--", color=C_HURDLE, lw=1.6, label="best constant — true hurdle")
    ax.plot(xs, cmean, "-^", color=C_CMEAN, ms=5, lw=1.6, label="Constant (mean) — measured")
    ax.plot(xs, gru, "-s", color=C_GRU, ms=5, lw=1.6, label="GRU")
    ax.plot(xs, prop, "-o", color=C_PROP, ms=6, lw=2.0, label="SE-ResNet-BiGRU (proposed)")

    ax2.plot(xs, li, color=C_LI, lw=1.2, alpha=0.8)
    ax2.plot(xs, li, "D", color=C_LI, ms=3.5, alpha=0.9)
    ax2.set_ylabel("label informativeness LI", color=C_LI)
    ax2.tick_params(axis="y", colors=C_LI)
    ax2.set_ylim(0, 1.05)

    # crossover: where proposed crosses the strongest trivial baseline
    hurdle = np.where(np.isnan(cmean), best, np.minimum(cmean, best))
    y = prop - hurdle
    xc = None
    for i in range(len(xs) - 1):
        if np.sign(y[i]) != np.sign(y[i + 1]):
            xc = xs[i] + (xs[i + 1] - xs[i]) * (-y[i]) / (y[i + 1] - y[i])
            break
    if xc is not None:
        ax.axvline(xc, color="#444444", lw=1.0, ls="-.")
        ax.annotate(f"crossover\n$h\\approx{xc:.2f}$", xy=(xc, ax.get_ylim()[1] * 0.72),
                    xytext=(xc + 0.06, ax.get_ylim()[1] * 0.80), fontsize=8, color="#444444")

    ax.set_xlabel("plateau length $h$  (0 = full-life linear label)")
    ax.set_ylabel("RMSE (min, 3-condition mean)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_title("Model verdict is decided by the label, not the architecture")
    h1, l1 = ax.get_legend_handles_labels()
    ax.legend(h1, l1, frameon=False, loc="upper left", ncol=1)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig2_spectrum.{ext}")
    plt.close(fig)


# --------------------------------------------------------------------- fig 3
def fig3() -> None:
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # --- (a) XJTU-SY, minutes
    xj_root = S5 / "single_linear"
    xj_anch = analytic_anchors(REPO / "artifacts" / "data_linear")
    conds = ["cond1", "cond2", "cond3"]
    series = {"se_resnet_bigru": C_PROP, "gru": C_GRU, "pure_bigru": C_BIGRU,
              "constant_mean": C_CMEAN, "constant_one": C_ONE}
    _grouped(ax[0], "a", xj_root, conds, series, xj_anch, "RMSE on XJTU-SY (min)", 1.0)

    # --- (b) PHM 2012, hours
    ph_root = S5 / "phm2012_single_linear"
    ph_anch = analytic_anchors(REPO / "artifacts" / "phm2012" / "data_linear")
    ph_conds = ["cond1", "cond2"]
    series_ph = {"se_resnet_bigru": C_PROP, "constant_mean": C_CMEAN, "constant_one": C_ONE}
    _grouped(ax[1], "b", ph_root, ph_conds, series_ph, ph_anch, "RMSE on PHM 2012 (h)", 1.0 / 3600.0)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig3_main_results.{ext}")
    plt.close(fig)


def _grouped(ax, letter: str, root: pathlib.Path, conds, series, anch, ylabel, scale) -> None:
    models = [m for m in series if (root / m / "results_corrected.csv").exists()]
    n = len(models)
    width = 0.8 / max(n, 1)
    x = np.arange(len(conds))
    for i, m in enumerate(models):
        data = cond_rmse(root, m)
        vals = [data.get(c, np.nan) * scale for c in conds]
        ax.bar(x + (i - (n - 1) / 2) * width, vals, width,
               color=series[m], label=MODEL_LABEL.get(m, m), edgecolor="white", linewidth=0.4)
    hv = [anch.get(c, {}).get("best_const", np.nan) * scale for c in conds]
    ax.plot(x, hv, "D", color="black", ms=6, label="best constant (hurdle)", zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(conds)
    ax.set_ylabel(ylabel)
    ax.set_title(f"({letter}) {ylabel.split(' on ')[-1].split(' (')[0]}")
    ax.legend(frameon=False, fontsize=7, loc="upper left")


def main() -> int:
    fig1()
    fig2()
    fig3()
    for p in sorted(OUT.glob("*.png")):
        print("wrote", p.relative_to(REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
