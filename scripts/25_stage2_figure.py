"""Stage-2 headline figure: the label-structure gap, in both calibers.

Plots, against the plateau length h (linear endpoint at h=0):

* ``gap vs best_const``  = model RMSE - best-constant RMSE  (post-hoc hurdle)
* ``Delta RMSE vs const_mean`` = model RMSE - deployable zero-learning baseline

for the **three-bearing mean** caliber and the **pooled** caliber. The zero
crossing -- where "learning" stops paying off -- is the Stage-2 headline.

    python scripts/25_stage2_figure.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    data = json.loads((REPO / "artifacts" / "reports" / "stage2_chain.json").read_text(encoding="utf-8"))
    rows = [r for r in data["rows"] if "chain_three_bearing" in r]

    h = [r["h"] for r in rows]
    li_cont = [r["LI_continuous"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    specs = [
        ("chain_three_bearing", "three-bearing mean"),
        ("chain_pooled", "pooled (n_windows-weighted)"),
    ]
    for ax, (key, title) in zip(axes, specs):
        gap = [r[key]["gap_vs_best_const"] for r in rows]
        dlt = [r[key]["delta_rmse_min"] for r in rows]
        ax.axhline(0.0, color="#888888", lw=1.0, ls=":")
        ax.plot(h, gap, "o-", color="#1f4e79", lw=2.0, ms=6, label=r"model $-$ best-const (post-hoc)")
        ax.plot(h, dlt, "s--", color="#c00000", lw=2.0, ms=6, label=r"$\Delta$RMSE vs const-mean (deployable)")
        ax.set_title(f"{title} caliber", fontsize=11)
        ax.set_xlabel("plateau length  h   (0.0 = linear endpoint)", fontsize=10)
        ax.set_xticks([0.0, 0.5, 0.6, 0.7, 0.8, 0.9])
        ax.grid(alpha=0.25, lw=0.6)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=8, loc="upper left", frameon=False)
    axes[0].set_ylabel("RMSE difference  [minutes]", fontsize=10)

    fig.suptitle("Stage 2 — sign of the learning gain flips as the label plateau grows",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = REPO / "paper" / "figures" / "fig_stage2_label_gap.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")

    # analytic vs measured LI, printed for the caption
    print("h, LI_continuous, LI_windowed(3-bearing), LI_pooled")
    for r in rows:
        print(f"  {r['h']:.1f}  {r['LI_continuous']:.3f}  "
              f"{r['anchors']['three_bearing']['LI']:.3f}  {r['anchors']['pooled']['LI']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
