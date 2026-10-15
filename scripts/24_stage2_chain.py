"""Stage-2 evidence chain:  h  ->  LI  ->  const_mean  ->  best_const  ->  RMSE  ->  G.

For every label point (the linear endpoint plus the piecewise plateaus
h in {0.5..0.9}) this assembles one row of the Stage-2 chain and writes both a
JSON and a markdown table.

Two halves, computed independently:

* **analytic** (no training, from the stored test splits): the zero-learning
  ladder ``const_one`` (ratio == 1.0), ``best_const`` (the post-hoc optimal
  constant, the *diagnostic* hurdle), ``const_zero``, and the label-structure
  index ``LI``.
* **measured** (from the Stage-2 report dirs under ``artifacts/reports/s2_chain``):
  the RMSE of every model actually trained under that label.

Discipline frozen in ``audit/STAGE1_PROTOCOL_FREEZE.md`` / ``audit/audit_01_LI.md``:

* statistics unit for the label spectrum is the **test bearing instance**
  -- ``(condition, bearing)``, i.e. exactly three groups in the single protocol
  (1_5, 2_5, 3_5). NOTE ``bearing_id`` alone is *not* a key: it is 5 in every
  condition, so grouping on it silently merges the three conditions.
* ``LI`` must be reported under **two calibers**: the continuous analytic
  ``3(1-h)/4`` and the **windowed measured three-bearing arithmetic mean**
  (the convention of ``09_label_baseline_anchors.py``; h=0.8 -> 0.247).
* ``best_const`` is **post-hoc only** -- never a deployable baseline.
* ``G = 1 - RMSE_model / RMSE_const_mean`` must always be reported next to the
  absolute ``Delta RMSE`` in minutes.

Both the **three-condition mean** caliber (the manuscript convention, matching
``summary_by_model.csv``) and the **pooled** caliber (n_windows-weighted) are
emitted, side by side.

Usage::

    python scripts/24_stage2_chain.py
    python scripts/24_stage2_chain.py --json artifacts/reports/stage2_chain.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

#: (display, data_dir, report_dir, continuous_h)  -- h=None for the linear endpoint
POINTS = [
    ("linear", "artifacts/data_linear", "artifacts/reports/s2_chain/linear", 0.0),
    ("h=0.5", "artifacts/data_h0.5", "artifacts/reports/s2_chain/h0.5", 0.5),
    ("h=0.6", "artifacts/data_h0.6", "artifacts/reports/s2_chain/h0.6", 0.6),
    ("h=0.7", "artifacts/data_h0.7", "artifacts/reports/s2_chain/h0.7", 0.7),
    ("h=0.8", "artifacts/data_h0.8", "artifacts/reports/s2_chain/h0.8", 0.8),
    ("h=0.9", "artifacts/data_h0.9", "artifacts/reports/s2_chain/h0.9", 0.9),
]
MODEL = "se_resnet_bigru"


# ------------------------------------------------------------------ analytic
def _test_groups(data_dir: pathlib.Path) -> dict:
    """Group the single-protocol test windows by (condition, bearing_id)."""
    index = json.loads((data_dir / "index.json").read_text(encoding="utf-8"))
    groups: dict = {}
    for rec in index:
        if rec.get("protocol") != "single":
            continue
        with np.load(rec["file"]) as d:
            if "test/ratio" not in d.files:
                continue
            r = np.asarray(d["test/ratio"], dtype=np.float64)
            life = np.asarray(d["test/lifetime"], dtype=np.float64)
            bid = np.asarray(d["test/bearing_id"])
        for b in sorted(set(bid.tolist())):
            m = bid == b
            groups[(rec.get("name", "?"), int(b))] = (r[m], life[m])
    return groups


def _li(r: np.ndarray) -> float:
    """LI = 1 - Var[r]/E[(r-1)^2]  (ratio space, unweighted)."""
    mse_one = float(np.mean((r - 1.0) ** 2))
    return 1.0 - float(np.var(r)) / mse_one if mse_one else float("nan")


def analytic_anchors(data_dir: pathlib.Path) -> dict:
    """Zero-learning ladder + LI, in BOTH calibers."""
    groups = _test_groups(data_dir)
    all_r = np.concatenate([r for r, _ in groups.values()])
    all_life = np.concatenate([l for _, l in groups.values()])

    # --- pooled caliber (window-level, minute-space RMSE)
    c_star = float(np.sum(all_life**2 * all_r) / np.sum(all_life**2))  # L^2 optimum
    pooled = {
        "const_one": float(np.sqrt(np.mean((all_life * (all_r - 1.0)) ** 2))),
        "const_zero": float(np.sqrt(np.mean((all_life * all_r) ** 2))),
        "best_const": float(np.sqrt(np.mean((all_life * (all_r - c_star)) ** 2))),
        "c_star": c_star,
        "LI": _li(all_r),
    }

    # --- three-bearing caliber (score each test bearing alone, then simple mean)
    one_b, best_b, li_b = [], [], []
    for r, life in groups.values():
        one_b.append(float(np.sqrt(np.mean((life * (r - 1.0)) ** 2))))
        best_b.append(float(np.sqrt(np.mean((life * (r - r.mean())) ** 2))))
        li_b.append(_li(r))
    bear = {
        "const_one": float(np.mean(one_b)),
        "best_const": float(np.mean(best_b)),
        "LI": float(np.mean(li_b)),
        "n_groups": len(one_b),
        "per_group": {
            f"{k[0]}::{k[1]}": {"const_one": one_b[i], "best_const": best_b[i], "LI": li_b[i]}
            for i, k in enumerate(sorted(groups))
        },
    }
    return {
        "n_windows": int(all_r.size),
        "n_bearings": len(groups),
        "pooled": pooled,
        "three_bearing": bear,
    }


# ------------------------------------------------------------------ measured
def measured(report_dir: pathlib.Path) -> dict:
    """Both RMSE calibers per model from a Stage-2 run's per-bearing block."""
    full = report_dir / "results_full.json"
    if not full.exists():
        return {}
    results = json.loads(full.read_text(encoding="utf-8"))
    agg: dict = {}
    for res in results:
        per_b = res.get("per_bearing") or []
        if not per_b:
            continue
        acc = agg.setdefault(res["model"], {"per_condition": {}})
        n_b = np.asarray([row["n_windows"] for row in per_b], dtype=float)
        r_b = np.asarray([row["rmse"] for row in per_b], dtype=float)
        # one condition == one test bearing in the single protocol
        acc["per_condition"][res["dataset"]] = float(np.sqrt(np.mean(r_b**2)))
        acc.setdefault("n", []).extend(n_b.tolist())
        acc.setdefault("rmse", []).extend(r_b.tolist())

    out = {}
    for model, acc in agg.items():
        n = np.asarray(acc["n"], dtype=float)
        rmse = np.asarray(acc["rmse"], dtype=float)
        per_cond = np.asarray(list(acc["per_condition"].values()), dtype=float)
        out[model] = {
            "three_cond_mean_rmse": float(per_cond.mean()),
            "pooled_rmse": float(np.sqrt(np.sum(n * rmse**2) / np.sum(n))),
            "per_condition": acc["per_condition"],
        }
    return out


# ----------------------------------------------------------------------- main
def build() -> list[dict]:
    rows = []
    for display, dd, rd, h in POINTS:
        anch = analytic_anchors(REPO / dd)
        meas = measured(REPO / rd)
        m, cm, co = meas.get(MODEL), meas.get("constant_mean"), meas.get("constant_one")
        row = {
            "label": display,
            "h": h,
            "LI_continuous": 3.0 * (1.0 - h) / 4.0,
            "anchors": anch,
            "measured": meas,
        }
        if m and cm:
            row["chain_three_bearing"] = {
                "rmse_model": m["three_cond_mean_rmse"],
                "rmse_const_mean": cm["three_cond_mean_rmse"],
                "rmse_const_one": co["three_cond_mean_rmse"] if co else None,
                "best_const": anch["three_bearing"]["best_const"],
                "G": 1.0 - m["three_cond_mean_rmse"] / cm["three_cond_mean_rmse"],
                "delta_rmse_min": m["three_cond_mean_rmse"] - cm["three_cond_mean_rmse"],
                "gap_vs_best_const": m["three_cond_mean_rmse"] - anch["three_bearing"]["best_const"],
                "LI": anch["three_bearing"]["LI"],
            }
            row["chain_pooled"] = {
                "rmse_model": m["pooled_rmse"],
                "rmse_const_mean": cm["pooled_rmse"],
                "rmse_const_one": co["pooled_rmse"] if co else None,
                "best_const": anch["pooled"]["best_const"],
                "G": 1.0 - m["pooled_rmse"] / cm["pooled_rmse"],
                "delta_rmse_min": m["pooled_rmse"] - cm["pooled_rmse"],
                "gap_vs_best_const": m["pooled_rmse"] - anch["pooled"]["best_const"],
                "LI": anch["pooled"]["LI"],
            }
        rows.append(row)
    return rows


def _f(v, w=9, p=2):
    return f"{'-':>{w}}" if v is None else f"{v:>{w}.{p}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(REPO / "artifacts" / "reports" / "stage2_chain.json"))
    ap.add_argument("--md", default=str(REPO / "docs" / "stage2_chain_table.md"))
    args = ap.parse_args()

    rows = build()

    print(f"model={MODEL}; baselines constant_one / constant_mean / best_const(post-hoc)")
    for caliber in ("three_bearing", "pooled"):
        key = f"chain_{caliber}"
        hdr = (f"[{caliber:>13}] {'label':>7} {'LI_cont':>8} {'LI_win':>7} {'c1':>8} "
               f"{'c_mean':>8} {'bc':>8} {'model':>8} {'G':>7} {'dRMSE':>8} {'gap_bc':>8}")
        print("\n" + hdr)
        print("-" * len(hdr))
        for row in rows:
            x = row.get(key)
            if not x:
                continue
            a = row["anchors"]
            li_win = a["three_bearing"]["LI"] if caliber == "three_bearing" else a["pooled"]["LI"]
            c1 = a["three_bearing"]["const_one"] if caliber == "three_bearing" else a["pooled"]["const_one"]
            print(f"{'':>15} {row['label']:>7} {row['LI_continuous']:>8.3f} {li_win:>7.3f} "
                  f"{c1:>8.2f} {_f(x['rmse_const_mean'], 8)} {x['best_const']:>8.2f} "
                  f"{_f(x['rmse_model'], 8)} {_f(x['G'], 7, 3)} "
                  f"{_f(x['delta_rmse_min'], 8)} {_f(x['gap_vs_best_const'], 8)}")

    pathlib.Path(args.json).write_text(
        json.dumps({"model": MODEL, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.json}")

    # machine-generated table (the hand-written report is audit/STAGE2_EVIDENCE_CHAIN.md)
    md_path = pathlib.Path(args.md)
    lines = [
        "# Stage-2 evidence chain (generated by scripts/24_stage2_chain.py)",
        "",
        f"Model `{MODEL}`; baselines `constant_one` (naive), `constant_mean` (deployable), "
        "`best_const` (**post-hoc only**). RMSE in minutes.",
        "",
        "`LI_cont` = continuous analytic `3(1-h)/4`; `LI_win` = windowed measured "
        "three-bearing arithmetic mean; `LI_pool` = window-pooled.",
        "",
    ]
    for caliber, title in (("three_bearing", "three-bearing mean (primary)"),
                           ("pooled", "pooled (n_windows-weighted)")):
        lines += [
            f"## {title}",
            "",
            "| label | h | LI_cont | LI_win | const_one | const_mean | best_const | model | G | dRMSE | model-best_const |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for row in rows:
            x = row.get(f"chain_{caliber}")
            if not x:
                continue
            a = row["anchors"]
            li_win = a["three_bearing"]["LI"] if caliber == "three_bearing" else a["pooled"]["LI"]
            c1 = a["three_bearing"]["const_one"] if caliber == "three_bearing" else a["pooled"]["const_one"]
            lines.append(
                f"| {row['label']} | {row['h']:.1f} | {row['LI_continuous']:.3f} | {li_win:.3f} | "
                f"{c1:.2f} | {x['rmse_const_mean']:.2f} | {x['best_const']:.2f} | {x['rmse_model']:.2f} | "
                f"{x['G']:+.3f} | {x['delta_rmse_min']:+.2f} | {x['gap_vs_best_const']:+.2f} |")
        lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
