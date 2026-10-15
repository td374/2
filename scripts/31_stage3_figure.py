"""Stage 3 figure: R_cross before vs after equalising the training volume.

Panel A  paired dumbbell -- one row per (fold, target) unit (n = 15), an open
         marker for the unmatched arm and a filled marker for the volume-matched
         arm, with a vertical line at R_cross = 1 (the null).
Panel B  the same numbers stratified by target condition, to show whether the
         target-3 excess (whose cross arms saw only 0.07-0.27x the training data)
         collapses once the volume is matched.

Reads ``artifacts/reports/vm_vs_unmatched.json`` (from ``scripts/30``).
English axis labels -- journal deliverable.
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "artifacts/reports/vm_vs_unmatched.json"
OUT = ROOT / "paper/figures/fig_stage3_vm_rcross.png"

TCOLOR = {"1": "#1f77b4", "2": "#ff7f0e", "3": "#2ca02c"}


def main() -> int:
    data = json.loads(SRC.read_text(encoding="utf-8"))
    rows = data["units"]
    rows.sort(key=lambda r: (r["target"], r["fold"]))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.0), gridspec_kw={"width_ratios": [1.55, 1]})

    # ---------------------------------------------------------------- panel A
    ax = axes[0]
    ys = range(len(rows))
    labels = []
    for y, r in zip(ys, rows):
        c = TCOLOR[r["target"]]
        ax.plot([r["R_un"], r["R_vm"]], [y, y], color=c, lw=1.4, alpha=0.55, zorder=1)
        ax.plot(r["R_un"], y, marker="o", mfc="white", mec=c, mew=1.6, ms=7, zorder=3)
        ax.plot(r["R_vm"], y, marker="o", mfc=c, mec=c, ms=7, zorder=3)
        labels.append(f"{r['fold']}, c{r['target']}")
    ax.axvline(1.0, color="0.35", ls="--", lw=1.2, zorder=0)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(r"$R_{cross}=RMSE_{cross}/RMSE_{single}$", fontsize=10)
    ax.set_ylabel("unit = (fold, target condition)", fontsize=10)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
    ax.set_xticklabels(["0.5", "0.7", "1.0", "1.5", "2.0", "3.0"])
    ax.set_title("A. Per-unit $R_{cross}$: open = unmatched, filled = volume-matched", fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    for t in ("1", "2", "3"):
        ax.plot([], [], marker="o", ls="-", color=TCOLOR[t], label=f"cond{t}")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    # ---------------------------------------------------------------- panel B
    ax = axes[1]
    targets = sorted({r["target"] for r in rows})
    width = 0.36
    for i, t in enumerate(targets):
        un = [r["R_un"] for r in rows if r["target"] == t]
        vm = [r["R_vm"] for r in rows if r["target"] == t]
        xs_un = i - width / 2
        xs_vm = i + width / 2
        ax.scatter([xs_un] * len(un), un, marker="o", facecolors="white",
                   edgecolors=TCOLOR[t], s=34, linewidths=1.4, zorder=3)
        ax.scatter([xs_vm] * len(vm), vm, marker="o", color=TCOLOR[t], s=34, zorder=3)
        for a, b in zip(un, vm):
            ax.plot([xs_un, xs_vm], [a, b], color="0.6", lw=0.9, alpha=0.7, zorder=1)
    ax.axhline(1.0, color="0.35", ls="--", lw=1.2, zorder=0)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([f"cond{t}" for t in targets], fontsize=9)
    ax.set_ylabel(r"$R_{cross}$", fontsize=10)
    ax.set_yscale("log")
    ax.set_yticks([0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
    ax.set_yticklabels(["0.5", "0.7", "1.0", "1.5", "2.0", "3.0"])
    ax.set_title("B. By target condition (left bar = unmatched, right = matched)", fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
