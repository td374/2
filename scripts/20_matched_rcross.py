"""Matched cross/single gain ratio R_cross, with a run-to-run noise band.

What this answers
-----------------
IP5 (reframed) asks a *discriminative* question, not a promotional one:

    On an identical test bearing, does training on other conditions change the
    error relative to training on the target condition's own data -- and is any
    such change larger than the pipeline's own run-to-run spread?

    R_cross = RMSE_cross / RMSE_single        (same test set, only the training
                                               condition differs)
    delta   = (R_cross - 1) * 100 %

    R_cross < 1  -> cross-condition training helps
    R_cross = 1  -> no detectable change
    R_cross > 1  -> cross-condition training hurts

Pairing rules (why the numbers are comparable at all)
-----------------------------------------------------
The matched protocol makes ``test`` a single bearing of the TARGET condition in
both arms, so ``R_cross`` needs no rescaling. Concretely, for a task
``src -> tgt`` the matched pair is

    cross  : train on ``src`` bearings (config ``splits.cross``), test tgt_b
    single : train on ``tgt`` bearings (config ``splits.single``), test tgt_b

with ``tgt_b`` identical in both. If a legacy cross run tested all five
bearings, its aggregate is dominated by long-life units and must NOT be used
here -- see ``docs/matched_cross_vs_incondition.md``.

Two metric flavours, per disclosure rule A4
-------------------------------------------
``uncorrected`` is the honest ladder number; ``corrected`` is the paper's
headline (median-ratio correction). After correction every constant predictor
collapses to ``median(val_ratio) * lifetime`` regardless of its value, so the
constant ladder only exists in the ``uncorrected`` block. Both are reported.

Noise band (rule A7)
--------------------
Pass ``--cross-dir`` more than once (e.g. the three probe repeats). The script
then reports, per cell, the spread of ``R_cross`` across the repeats. Any cell
whose signal is smaller than this spread is not claimable; the summary prints
the median per-cell repeat spread next to the median ``R_cross``.

Pure re-analysis: reads files already on disk, trains nothing.

Usage
-----
    # pilot (single cross run)
    python scripts/20_matched_rcross.py

    # with the pilot probe's three repeats as a noise band
    python scripts/20_matched_rcross.py \\
        --cross-dir artifacts/repro_matched/repeat1 \\
        --cross-dir artifacts/repro_matched/repeat2 \\
        --cross-dir artifacts/repro_matched/repeat3

    # five-fold aggregate (one dir per fold)
    python scripts/20_matched_rcross.py \\
        --single-dir artifacts/reports/matched_fold1/single \\
        --cross-dir  artifacts/reports/matched_fold1/cross
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
DEFAULT_CONSTANTS = ["constant_one", "constant_mean", "ridge"]


# --------------------------------------------------------------------- loading
def _rows(path: pathlib.Path, metric: str):
    """Read one results_{metric}.csv; accept the corrected/uncorrected spelling."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _glob_metric(d: pathlib.Path, metric: str):
    """Every results_<metric>.csv under ``d`` (top level or per-model subdir)."""
    if not d.exists():
        return []
    hits = sorted(d.glob(f"results_{metric}.csv")) + sorted(d.glob(f"*/results_{metric}.csv"))
    return hits


def load_side(d: pathlib.Path, metric: str) -> dict[tuple[str, str], float]:
    """{(dataset, model): rmse} pooled over every table found under ``d``."""
    out: dict[tuple[str, str], float] = {}
    for path in _glob_metric(d, metric):
        for r in _rows(path, metric):
            ds, model, rmse = r.get("dataset"), r.get("model"), r.get("rmse")
            if not ds or not model or not rmse:
                continue
            try:
                out[(ds, model)] = float(rmse)
            except ValueError:
                continue
    return out


# ----------------------------------------------------------------- task helpers
def parse_task(task: str):
    """'1+2->3' -> (['1','2'], '3'); '1->2' -> (['1'], '2')."""
    if "->" not in task:
        return None
    src, tgt = task.split("->")
    return src.split("+"), tgt


def cond_of_single(dataset: str):
    """'cond2' -> '2'."""
    return dataset[4:] if dataset.startswith("cond") else None


# ------------------------------------------------------------------ statistics
def median(xs):
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
    lo = xs[n // 4]
    hi = xs[(3 * n) // 4]
    return hi - lo


def fmt(x, nd=2):
    if x is None:
        return "-"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return f"{v:.{nd}f}" if math.isfinite(v) else "-"


# ------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--single-dir", default="artifacts/reports/single_linear",
                    help="in-condition report dir (may be per-model subdirs)")
    ap.add_argument("--cross-dir", action="append", default=None,
                    help="matched cross report dir; repeat for a noise band")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--constants", nargs="+", default=DEFAULT_CONSTANTS)
    ap.add_argument("--out", default="docs/matched_rcross.md")
    ap.add_argument("--json", default="artifacts/reports/matched_rcross.json")
    args = ap.parse_args()

    cross_dirs = [ROOT / p for p in (args.cross_dir or ["artifacts/reports/cross_matched_linear"])]
    single_dir = ROOT / args.single_dir

    print("single arm:", single_dir.relative_to(ROOT) if single_dir.is_absolute() else single_dir)
    print("cross arms:", [str(p.relative_to(ROOT)) for p in cross_dirs])

    # single_map[(cond, model)] = rmse ; single datasets are named cond1..cond3
    single = {}
    for metric in ("uncorrected", "corrected"):
        m = load_side(single_dir, metric)
        single[metric] = {(cond_of_single(ds), model): v
                          for (ds, model), v in m.items() if cond_of_single(ds)}
    have_single = single["uncorrected"]
    if not have_single:
        print(f"\n!! no single-condition tables found under {single_dir}")
        print("   pass --single-dir (e.g. artifacts/reports/single_linear)")
        return 2

    # cross[i][metric][(task, model)] = rmse
    cross = []
    for d in cross_dirs:
        per = {}
        for metric in ("uncorrected", "corrected"):
            per[metric] = load_side(d, metric)
        cross.append({"dir": d, "map": per})
        n_tasks = len({t for (t, _m) in per["uncorrected"]})
        print(f"  {d.relative_to(ROOT)}: tasks={n_tasks} cells={len(per['uncorrected'])}")

    if not any(c["map"]["uncorrected"] for c in cross):
        print("\n!! no cross tables found (probe still running? wrong --cross-dir?)")
        return 2

    # collect the cells (task, model) available in the first non-empty cross arm
    first = next(c for c in cross if c["map"]["uncorrected"])
    cells = sorted(first["map"]["uncorrected"])
    tasks = sorted({t for (t, _m) in cells})
    print(f"\ntasks ({len(tasks)}): {tasks}")

    report: dict = {"single_dir": str(single_dir), "cross_dirs": [str(c["dir"]) for c in cross],
                    "metrics": {}}

    lines: list[str] = []
    J = lines.append

    J("# Matched cross/single gain ratio `R_cross`")
    J("")
    J("`R_cross = RMSE_cross / RMSE_single` measured on an **identical test bearing**:")
    J("only the condition the training data comes from changes. Regenerate with")
    J("`python scripts/20_matched_rcross.py`.")
    J("")
    J(f"- in-condition arm: `{single_dir.relative_to(ROOT)}`")
    for c in cross:
        try:
            rel = c["dir"].relative_to(ROOT)
        except ValueError:
            rel = c["dir"]
        J(f"- cross arm: `{rel}`")
    J("")
    if len(cross) > 1:
        J(f"> {len(cross)} cross arms supplied: the per-cell spread across them is the")
        J("> pipeline's run-to-run noise (**rule A7**). A cell whose `R_cross` moves less")
        J("> than this spread carries no claim.")
        J("")

    for metric in ("uncorrected", "corrected"):
        J(f"## Metric: `{metric}`" + ("  (paper headline)" if metric == "corrected" else "  (honest ladder)"))
        J("")
        if metric == "corrected":
            J("> After correction every constant collapses to `median(val_ratio) * lifetime`,")
            J("> so the constant ladder is only visible in the `uncorrected` block (rule A4).")
            J("")

        # ---- per (target, model): best-source R_cross, and the full cell range
        J("| target | model | RMSE_single | best RMSE_cross (task) | R_cross best | R_cross median | median dRMSE (min) | cells <1 | range |")
        J("|---|---|---|---|---|---|---|---|---|")
        block = {}
        targets = sorted({parse_task(t)[1] for t in tasks if parse_task(t)})
        for tgt in targets:
            for model in args.models:
                inc = single[metric].get((tgt, model))
                per_task = {}
                for task in tasks:
                    pt = parse_task(task)
                    if not pt or pt[1] != tgt:
                        continue
                    # one value per cross arm
                    vals = []
                    for c in cross:
                        v = c["map"][metric].get((task, model))
                        if v is not None:
                            vals.append(v)
                    if vals:
                        per_task[task] = vals
                if inc is None or not per_task:
                    continue
                # best source = smallest mean cross RMSE across arms
                best_task = min(per_task, key=lambda t: statistics.fmean(per_task[t]))
                best_mean = statistics.fmean(per_task[best_task])
                r_best = best_mean / inc if inc else float("nan")
                # per-cell R across arms -> cell medians
                cell_R = {t: [v / inc for v in vs] for t, vs in per_task.items() if inc}
                cell_med = [median(vs) for vs in cell_R.values()]
                # per-cell absolute difference in minutes (rule D4/C6): Delta = RMSE_cross - RMSE_single
                cell_D = {t: [v - inc for v in vs] for t, vs in per_task.items()}
                med_D = median([median(vs) for vs in cell_D.values()]) if cell_D else float("nan")
                n_better = sum(1 for r in cell_med if r < 1.0)
                rng = (max(cell_med) - min(cell_med)) if cell_med else float("nan")
                J(f"| {tgt} | {model} | {fmt(inc)} | {fmt(best_mean)} ({best_task}) | "
                  f"**{fmt(r_best, 3)}** | {fmt(median(cell_med), 3)} | {med_D:+.2f} | "
                  f"{n_better}/{len(cell_med)} | {fmt(rng, 2)} |")
                block[f"{tgt}|{model}"] = {
                    "rmse_single": inc,
                    "r_cross_best": r_best,
                    "r_cross_cells": {t: vs for t, vs in cell_R.items()},
                    "delta_rmse_cells": {t: vs for t, vs in cell_D.items()},
                    "median_cell_R": median(cell_med),
                    "median_delta_rmse": med_D,
                    "cells_below_one": n_better,
                    "n_cells": len(cell_med),
                }

        # ---- summary over every (target, model, task) cell
        all_R, all_D = [], []
        for key, b in block.items():
            for t, vs in b["r_cross_cells"].items():
                all_R.extend(vs)
            for t, vs in b.get("delta_rmse_cells", {}).items():
                all_D.extend(vs)
        if all_R:
            med = median(all_R)
            q = iqr(all_R)
            win = sum(1 for r in all_R if r < 1.0) / len(all_R)
            md = median(all_D) if all_D else float("nan")
            J("")
            J(f"**Summary ({metric})** over all {len(all_R)} (target, model, source-task, repeat) cells: "
              f"median `R_cross` = **{med:.3f}**, IQR = {q:.3f}, "
              f"median `dRMSE` = **{md:+.2f} min**, "
              f"cross better in {100 * win:.0f}%, range {min(all_R):.3f}-{max(all_R):.3f}.")
        report["metrics"][metric] = {"cells": block, "all_R": all_R,
                                     "median_delta_rmse": (median(all_D) if all_D else None)}

        # ---- repeat spread per cell (rule A7 noise band)
        if len(cross) > 1:
            spreads = []
            worst = []
            for key, b in block.items():
                for t, vs in b["r_cross_cells"].items():
                    if len(vs) < 2:
                        continue
                    mu = statistics.fmean(vs)
                    if mu:
                        sp = 100.0 * (max(vs) - min(vs)) / mu
                        spreads.append(sp)
                        worst.append((sp, key, t, vs))
            if spreads:
                J("")
                J(f"### Run-to-run spread of `R_cross` over {len(cross)} identical repeats")
                J("")
                J(f"median per-cell spread: **{median(spreads):.1f}%**, "
                  f"max: {max(spreads):.1f}%.")
                worst.sort(reverse=True)
                J("")
                J("| cell (target\\|model) | task | R_cross per repeat | spread |")
                J("|---|---|---|---|")
                for sp, key, t, vs in worst[:6]:
                    J(f"| {key} | {t} | {' / '.join(fmt(v, 3) for v in vs)} | {sp:.1f}% |")
        J("")

    # ---- baseline ladder on the matched test bearing
    J("## Baseline ladder (matched test bearing, `uncorrected`)")
    J("")
    J("Same test bearing as `R_cross`. `constant_*` are training-condition independent,")
    J("so their error is the floor any model must beat to be non-trivial (rule A2).")
    J("")
    J("| target | model | RMSE (single arm) | RMSE (cross arm, best task) |")
    J("|---|---|---|---|")
    for tgt in ["1", "2", "3"]:
        for model in args.constants + args.models:
            inc = single["uncorrected"].get((tgt, model))
            vals = []
            for c in cross:
                for (task, m), v in c["map"]["uncorrected"].items():
                    pt = parse_task(task)
                    if m == model and pt and pt[1] == tgt:
                        vals.append(v)
            cx = min(vals) if vals else None
            J(f"| {tgt} | {model} | {fmt(inc)} | {fmt(cx)} |")
    J("")

    # ---- decision aid vs the noise band
    if len(cross) > 1 and report["metrics"]["uncorrected"]["all_R"]:
        J("## Decision aid")
        J("")
        spreads = []
        for b in report["metrics"]["uncorrected"]["cells"].values():
            for vs in b["r_cross_cells"].values():
                if len(vs) > 1:
                    mu = statistics.fmean(vs)
                    if mu:
                        spreads.append(100.0 * (max(vs) - min(vs)) / mu)
        med_spread = median(spreads) if spreads else float("nan")
        allR = report["metrics"]["uncorrected"]["all_R"]
        sig = [abs(r - 1.0) * 100.0 for r in allR]
        frac_above = sum(1 for s in sig if s > med_spread) / len(sig) if sig else float("nan")
        J(f"- median run-to-run spread of `R_cross`: **{med_spread:.1f}%**")
        J(f"- median |R_cross - 1| (the 'signal'): **{median(sig):.1f}%**")
        J(f"- share of cells whose signal exceeds the noise band: **{100 * frac_above:.0f}%**")
        J("")
        J("Read-off: if the signal is comparable to the noise band, the correct IP5")
        J("conclusion is *'no stable direction / insufficient determinism'*, not 'cross")
        J("helps' or 'cross hurts'.")
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
