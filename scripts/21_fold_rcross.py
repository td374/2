"""Aggregate the leave-one-bearing-out folds into one R_cross distribution.

Relationship to scripts/20
--------------------------
``scripts/20_matched_rcross.py`` answers the PILOT question: one matched cross
run (optionally repeated) -> a run-to-run noise band for ``R_cross``.

This script answers the HIGH-POWER question the reviewer will actually ask:
across five leave-one-bearing-out folds, is the *direction* of the
cross-vs-single difference stable, or does it flip?

Each fold already yields a clean ratio, because inside a fold the in-condition
arm and the cross arm share a byte-identical test bearing (see
``scripts/19_make_matched_folds.py``):

    R_cross(fold, target, model, source-task)
        = RMSE_cross / RMSE_single        (same test bearing)

Pooling over {5 folds} x {3 targets} x {models} x {source-tasks} gives a
distribution of paired ratios. Only the *direction* and the *dispersion* are
claimable; individual cells are not (rule A7).

Read-off (the three legal IP5 conclusions -- no preset answer)
--------------------------------------------------------------
A) stable direction  : median R_cross departs from 1 in ONE direction, the win
                       rate is far from 50%, and the effect exceeds the fold
                       spread.
B) no stable direction: median R_cross ~ 1 and/or the sign flips across folds.
C) insufficient determinism: the fold-to-fold spread of R_cross is comparable
                       to |R_cross - 1| itself.

Usage
-----
    python scripts/21_fold_rcross.py \\
        --fold artifacts/reports/matched_fold1 \\
        --fold artifacts/reports/matched_fold2 \\
        --fold artifacts/reports/matched_fold3 \\
        --fold artifacts/reports/matched_fold4 \\
        --fold artifacts/reports/matched_fold5

    # an explicit non-standard layout
    python scripts/21_fold_rcross.py \\
        --fold artifacts/reports/f1/single=artifacts/reports/f1/cross

Trains nothing; reads ``results_{uncorrected,corrected}.csv`` only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_MODELS = ["se_resnet_bigru", "pure_bigru", "cnn", "lstm", "gru"]


# --------------------------------------------------------------------- loading
def load_side(d: pathlib.Path, metric: str) -> dict[tuple[str, str], float]:
    """{(dataset, model): rmse} from every results_<metric>.csv under ``d``."""
    out: dict[tuple[str, str], float] = {}
    if not d.exists():
        return out
    for path in sorted(d.glob(f"results_{metric}.csv")) + sorted(d.glob(f"*/results_{metric}.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                ds, model, rmse = r.get("dataset"), r.get("model"), r.get("rmse")
                if not ds or not model or not rmse:
                    continue
                try:
                    out[(ds, model)] = float(rmse)
                except ValueError:
                    continue
    return out


def parse_task(task: str):
    if "->" not in task:
        return None
    src, tgt = task.split("->")
    return src.split("+"), tgt


def cond_of_single(ds: str):
    return ds[4:] if ds.startswith("cond") else None


# ------------------------------------------------------------------ statistics
def med(xs):
    xs = sorted(x for x in xs if x is not None and math.isfinite(x))
    if not xs:
        return float("nan")
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def iqr(xs):
    xs = sorted(x for x in xs if x is not None and math.isfinite(x))
    if len(xs) < 4:
        return float("nan")
    n = len(xs)
    return xs[(3 * n) // 4] - xs[n // 4]


def fmt(x, nd=2):
    if x is None:
        return "-"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return f"{v:.{nd}f}" if math.isfinite(v) else "-"


def wilcoxon_signed(xs, mu=0.0):
    """Two-sided Wilcoxon signed-rank vs mu, or None if scipy is unavailable.

    Cells are NOT independent (neighbouring cells share a fold and a bearing),
    so treat the p-value as descriptive, not as a confirmatory test.
    """
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return None
    xs = [x for x in xs if x is not None and math.isfinite(x) and x != mu]
    if len(xs) < 6:
        return None
    try:
        return float(wilcoxon(xs, alternative="two-sided").pvalue)
    except Exception:
        return None


# ------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", action="append", required=True,
                    help="fold dir (expects DIR/single and DIR/cross), or 'SINGLE=CROSS'")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--metric", choices=["uncorrected", "corrected", "both"], default="both")
    ap.add_argument("--out", default="docs/matched_fold_rcross.md")
    ap.add_argument("--json", default="artifacts/reports/matched_fold_rcross.json")
    args = ap.parse_args()

    metrics = ["uncorrected", "corrected"] if args.metric == "both" else [args.metric]

    folds = []
    for spec in args.fold:
        if "=" in spec:
            s, c = spec.split("=", 1)
            single_dir, cross_dir = pathlib.Path(s), pathlib.Path(c)
        else:
            base = pathlib.Path(spec)
            single_dir, cross_dir = base / "single", base / "cross"
        folds.append({"name": spec, "single": single_dir, "cross": cross_dir})
        print(f"fold {spec}: single={single_dir} cross={cross_dir}")

    # load everything
    data = {}
    for metric in metrics:
        per_fold = []
        for f in folds:
            s = {(cond_of_single(ds), m): v for (ds, m), v in load_side(f["single"], metric).items()
                 if cond_of_single(ds)}
            c = load_side(f["cross"], metric)
            per_fold.append({"name": f["name"], "single": s, "cross": c})
            print(f"  [{metric}] {f['name']}: single cells={len(s)} cross cells={len(c)}")
        data[metric] = per_fold

    any_cross = any(len(f["cross"]) for metric in metrics for f in data[metric])
    if not any_cross:
        print("\n!! no cross tables found -- have the folds been run yet?")
        print("   run: python scripts/19_make_matched_folds.py --run --models se_resnet_bigru")
        return 2

    lines: list[str] = []
    J = lines.append
    J("# Leave-one-bearing-out folds: `R_cross` distribution")
    J("")
    J("Pooled over the five folds. Inside a fold the two arms share an identical")
    J("test bearing, so every `R_cross` is a clean paired ratio; across folds the")
    J("test bearing rotates, which is the point (no single bearing is special).")
    J("Regenerate with `python scripts/21_fold_rcross.py`.")
    J("")
    J(f"folds: {len(folds)}  |  " + ", ".join(f"`{f['name']}`" for f in folds))
    J("")

    report: dict = {"folds": [f["name"] for f in folds], "metrics": {}}

    for metric in metrics:
        per_fold = data[metric]
        J(f"## Metric: `{metric}`" + ("  (paper headline)" if metric == "corrected" else "  (honest ladder)"))
        J("")

        # cells[(target, model)][task] = [{"R": ratio, "d": delta_rmse_min} per fold]
        cells: dict = {}
        for f in per_fold:
            tasks = sorted({t for (t, _m) in f["cross"]})
            for task in tasks:
                pt = parse_task(task)
                if not pt:
                    continue
                _srcs, tgt = pt
                for model in args.models:
                    inc = f["single"].get((tgt, model))
                    cx = f["cross"].get((task, model))
                    if inc and cx and inc > 0:
                        cells.setdefault((tgt, model), {}).setdefault(task, []).append(
                            {"R": cx / inc, "d": cx - inc}
                        )

        if not cells:
            J("_(no overlapping in-condition / cross cells)_")
            J("")
            continue

        J("| target | model | `R_cross` (fold x task) | median R | median dRMSE (min) | cells <1 | IQR |")
        J("|---|---|---|---|---|---|---|")
        all_R, all_L, all_D = [], [], []
        for (tgt, model) in sorted(cells):
            recs = [rec for vs in cells[(tgt, model)].values() for rec in vs]
            flat = [rec["R"] for rec in recs]
            flat_d = [rec["d"] for rec in recs]
            for r in flat:
                if r > 0:
                    all_R.append(r)
                    all_L.append(math.log(r))
            all_D.extend(flat_d)
            n_better = sum(1 for r in flat if r < 1)
            J(f"| {tgt} | {model} | {' / '.join(fmt(r, 3) for r in flat)} | "
              f"**{fmt(med(flat), 3)}** | **{med(flat_d):+.2f}** | {n_better}/{len(flat)} | "
              f"{fmt(iqr(flat), 3)} |")

        J("")
        m = med(all_R)
        md = med(all_D)
        win = sum(1 for r in all_R if r < 1) / len(all_R)
        p = wilcoxon_signed(all_L, 0.0)
        J(f"**Pooled ({metric})**: {len(all_R)} paired ratios; median `R_cross` = **{m:.3f}** "
          f"(delta {100 * (m - 1):+.1f}%); median `dRMSE` = **{md:+.2f} min** "
          f"(paired absolute difference, rule D4/C6); cross better in {100 * win:.0f}%; "
          f"IQR = {iqr(all_R):.3f}; range {min(all_R):.3f}-{max(all_R):.3f}.")
        if p is not None:
            J(f"Wilcoxon signed-rank on `log R` vs 0: p = {p:.3g} (descriptive only -- cells "
              f"are not independent).")
        J("")

        # fold-to-fold stability of the *median*
        J("### Stability across folds")
        J("")
        J("| fold | n cells | median R_cross | share <1 |")
        J("|---|---|---|---|")
        fold_meds = []
        for f in per_fold:
            tasks = sorted({t for (t, _m) in f["cross"]})
            rs = []
            for task in tasks:
                pt = parse_task(task)
                if not pt:
                    continue
                _srcs, tgt = pt
                for model in args.models:
                    inc = f["single"].get((tgt, model))
                    cx = f["cross"].get((task, model))
                    if inc and cx and inc > 0:
                        rs.append(cx / inc)
            if rs:
                fold_meds.append(med(rs))
                J(f"| `{f['name']}` | {len(rs)} | {fmt(med(rs), 3)} | "
                  f"{100 * sum(1 for r in rs if r < 1) / len(rs):.0f}% |")
        if len(fold_meds) >= 2:
            spread = 100.0 * (max(fold_meds) - min(fold_meds)) / statistics.fmean(fold_meds)
            n_pos = sum(1 for x in fold_meds if x > 1.0)
            J("")
            J(f"fold-to-fold spread of the fold median: **{spread:.1f}%**")
            J("")
            sig = abs(m - 1.0) * 100.0
            J(f"compare: |median R_cross - 1| = **{sig:.1f}%** vs between-fold spread **{spread:.1f}%**")
            J(f"sign consistency: fold medians > 1 in **{n_pos}/{len(fold_meds)}** folds")
            J("")
            J("NOTE: run-to-run determinism is a *separate* question -- measure it with")
            J("`scripts/17_reproducibility_probe.py`. A large spread here means the effect")
            J("genuinely depends on **which bearing is held out**, not that the pipeline is noisy.")
            J("")
            if n_pos in (0, len(fold_meds)) and sig > spread:
                J("-> (A) stable direction: every fold lands on the same side and the effect")
                J("   exceeds the between-fold spread.")
            elif n_pos in (0, len(fold_meds)):
                J("-> (A') consistent sign but the effect is within the between-fold spread:")
                J("   report the direction with the interval, do not quote a single number.")
            else:
                J("-> (B) **no stable direction**: the fold medians fall on BOTH sides of 1,")
                J("   so the sign depends on which bearing is held out. Report the distribution.")
            J("")
        report["metrics"][metric] = {
            "pooled_median": m, "win_rate": win, "iqr": iqr(all_R),
            "median_delta_rmse_min": md, "n": len(all_R), "wilcoxon_p": p,
            "fold_medians": fold_meds,
            "cells": {f"{k[0]}|{k[1]}": {t: vs for t, vs in v.items()} for k, v in cells.items()},
        }
        J("")

    out_md = ROOT / args.out
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out_md}")

    out_json = ROOT / args.json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
