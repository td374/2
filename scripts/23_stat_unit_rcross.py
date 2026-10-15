"""Unit-level `R_cross` analysis under the frozen statistical-unit protocol.

Why this script exists
----------------------
``scripts/21_fold_rcross.py`` pooled every (fold, target, task, model) cell into one
distribution and reported a Wilcoxon p-value over it. That p-value is **not** a valid
confirmatory statistic: the cells are not independent. The frozen protocol
(``audit/STATISTICAL_UNIT_PROTOCOL.md``) fixes the unit and this script implements it.

The unit is the TEST BEARING INSTANCE
-------------------------------------
``configs/matched_folds/fold{k}.yaml`` makes fold ``k`` test bearing ``k``, and inside a
fold the in-condition and cross arms share a byte-identical test bearing. Unrolling the
five folds therefore visits

    5 folds x 3 target conditions = 15 test-bearing instances
                                  == all 15 XJTU-SY bearings

so **n = 15 independent units**. Each unit carries 3 source-task variants (two single-
source plus one multi-source) which are REPEATED MEASURES on the same bearing, not extra
samples. Hence:

    45 cells = 15 units x 3 repeated measures

What this script reports (protocol S4-S8)
-----------------------------------------
* unit-level distribution: median, IQR, min-max, win rate, sign consistency
* effect sizes: ``R_cross`` **and** ``delta_rmse`` in minutes (S5)
* bearing-level percentile bootstrap CI, resampling the 15 UNITS (S6)
* stratified by target condition, because the training-volume confound flips sign with
  the target (S8)
* the 45-cell pooled p is printed but labelled ``exploratory``, never as core evidence (S7)

Caveat printed in the output: even the 15 units share training bearings across folds, so
they are not strictly independent and the bootstrap CI is a LOWER bound on the true
uncertainty (S10).

Pure re-analysis. Trains nothing.

Usage
-----
    python scripts/23_stat_unit_rcross.py
    python scripts/23_stat_unit_rcross.py --models se_resnet_bigru pure_bigru
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import random
import re
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_MODELS = ["se_resnet_bigru"]
#: ``_b8`` = the M1 baseline re-evaluated with the current (post-B8) code, so the
#: numerator and denominator of ``R_cross`` come from the same evaluation vintage
#: (audit finding B9, see ``docs/stat_unit_rcross.md`` and freeze red line 23).
DEFAULT_FOLDS = [f"artifacts/reports/matched_fold{k}_b8" for k in range(1, 6)]

_EST_ADVICE = (
    "H4 evidence is the DISTRIBUTION + effect size + bearing-level CI, "
    "not any p-value."
)


# --------------------------------------------------------------------- loading
def load_results(d: pathlib.Path, metric: str) -> dict[tuple[str, str], float]:
    """{(dataset, model): rmse} from every results_<metric>.csv under ``d``."""
    out: dict[tuple[str, str], float] = {}
    if not d.exists():
        return out
    paths = sorted(d.glob(f"results_{metric}.csv")) + sorted(d.glob(f"*/results_{metric}.csv"))
    for path in paths:
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


def boot_ci(values, n_boot=10_000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for the median, resampling whole UNITS (protocol S6)."""
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if len(vals) < 4:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(vals)
    stats = []
    for _ in range(n_boot):
        stats.append(med([vals[rng.randrange(n)] for _ in range(n)]))
    stats.sort()
    lo = stats[int(alpha / 2 * n_boot)]
    hi = stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


def wilcoxon(xs, mu=0.0):
    """Two-sided Wilcoxon signed-rank; returns None when scipy is absent/too few."""
    try:
        from scipy.stats import wilcoxon as _w
    except Exception:
        return None
    vals = [x for x in xs if x is not None and math.isfinite(x) and x != mu]
    if len(vals) < 6:
        return None
    try:
        return float(_w(vals, alternative="two-sided").pvalue)
    except Exception:
        return None


def fmt(x, nd=3):
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
    ap.add_argument("--fold", action="append", default=None,
                    help="fold dir (expects DIR/single and DIR/cross); default = matched_fold1..5")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--metrics", nargs="+", default=["uncorrected", "corrected"],
                    choices=["uncorrected", "corrected"])
    ap.add_argument("--primary-metric", default="uncorrected",
                    help="frozen headline metric; the other is reported as sensitivity")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="docs/stat_unit_rcross.md")
    ap.add_argument("--json", default="artifacts/reports/stat_unit_rcross.json")
    args = ap.parse_args()

    fold_specs = args.fold if args.fold else DEFAULT_FOLDS
    folds = []
    for spec in fold_specs:
        base = pathlib.Path(spec)
        if not base.is_absolute():
            base = ROOT / base
        folds.append({"name": pathlib.Path(spec).name, "base": base})

    # ---- build the cell table: (metric, model) -> (target, task) -> [(fold, R, dRMSE)]
    table: dict = {}
    coverage = []
    for metric in args.metrics:
        per_model: dict = {}
        for f in folds:
            s = load_results(f["base"] / "single", metric)
            c = load_results(f["base"] / "cross", metric)
            n_single = sum(1 for (ds, m) in s if cond_of_single(ds) and m in args.models)
            n_cross = sum(1 for (td, m) in c if parse_task(td) and m in args.models)
            coverage.append({"fold": f["name"], "metric": metric,
                             "single_cells": n_single, "cross_cells": n_cross})
            for (ds, model), inc in s.items():
                if model not in args.models:
                    continue
                tgt = cond_of_single(ds)
                if tgt is None or not (inc and inc > 0):
                    continue
                for (task, m2), cx in c.items():
                    if m2 != model:
                        continue
                    pt = parse_task(task)
                    if not pt or pt[1] != tgt:
                        continue
                    per_model.setdefault((tgt, model), {}).setdefault(task, []).append(
                        {"fold": f["name"], "R": cx / inc, "dRMSE": cx - inc,
                         "rmse_single": inc, "rmse_cross": cx}
                    )
        table[metric] = per_model

    if not any(table.values()):
        print("!! no overlapping single/cross cells found -- have the folds been run?")
        print("   python scripts/19_make_matched_folds.py --run --models se_resnet_bigru")
        return 2

    lines: list[str] = []
    J = lines.append
    J("# Unit-level `R_cross` (frozen statistical-unit protocol)")
    J("")
    J("Statistical unit = **test-bearing instance** = `(fold, target condition)`, which is exactly")
    J("`(condition, bearing)`: 5 folds x 3 conditions = **15 units** = all 15 XJTU-SY bearings.")
    J("")
    J("The 3 source-task variants inside a unit are **repeated measures** on the same bearing,")
    J("pre-aggregated by median (protocol S2). So `45 cells = 15 units x 3 repeats`, and the")
    J("pooled cell-level p-value is **exploratory only** (protocol S7):")
    J("")
    J(f"> {_EST_ADVICE}")
    J("")
    J(f"primary metric: `{args.primary_metric}`  |  sensitivity: "
      + ", ".join(f"`{m}`" for m in args.metrics if m != args.primary_metric))
    J("")

    report: dict = {"primary_metric": args.primary_metric, "models": args.models,
                    "folds": [f["name"] for f in folds], "coverage": coverage, "metrics": {}}

    for metric in args.metrics:
        per_model = table[metric]
        tag = "  **(primary / headline)**" if metric == args.primary_metric else "  (sensitivity only)"
        J(f"## Metric: `{metric}`{tag}")
        J("")
        if metric == "corrected":
            J("> Reminder (frozen): the asymmetric median-ratio correction is **not** the primary")
            J("> evidence; it is reported for sensitivity and the constant ladder collapses in it.")
            J("")

        for model in args.models:
            keys = sorted(k for k in per_model if k[1] == model)
            if not keys:
                continue
            J(f"### Model: `{model}`")
            J("")
            J("| unit = (condition, bearing) | test bearing | tasks (repeats) | median R_cross | median dRMSE (min) |")
            J("|---|---|---|---|---|")
            units: list[dict] = []
            for tgt in sorted({k[0] for k in keys}):
                for task in sorted(per_model[(tgt, model)]):
                    for rec in per_model[(tgt, model)][task]:
                        units.append({"target": tgt, "fold": rec["fold"], "task": task,
                                      "R": rec["R"], "dRMSE": rec["dRMSE"]})
            by_unit: dict = {}
            for u in units:
                by_unit.setdefault((u["target"], u["fold"]), []).append(u)
            unit_rows = []
            for (tgt, fold) in sorted(by_unit):
                us = by_unit[(tgt, fold)]
                # fold names look like ``matched_fold3`` or ``matched_fold3_b8``;
                # take the FIRST digit run, not the last char (which is '8' in the
                # ``_b8`` vintage tag and would be mis-read as bearing 8).
                m = re.search(r"fold[_-]?(\d)", fold)
                bearing = m.group(1) if m else "?"
                R = med([u["R"] for u in us])
                dR = med([u["dRMSE"] for u in us])
                unit_rows.append({"target": tgt, "fold": fold, "bearing": f"b{bearing}",
                                  "n_tasks": len(us), "R": R, "dRMSE": dR,
                                  "tasks": [u["task"] for u in us]})
                J(f"| (cond{tgt}, b{bearing}) | b{bearing} | {len(us)} | "
                  f"{fmt(R)} | {dR:+.2f} |")
            J("")

            if not unit_rows:
                continue

            all_R = [u["R"] for u in unit_rows]
            all_dR = [u["dRMSE"] for u in unit_rows]
            cell_R = [u["R"] for u in units]
            cell_dR = [u["dRMSE"] for u in units]

            ci_R = boot_ci(all_R, n_boot=args.n_boot, seed=args.seed)
            ci_dR = boot_ci(all_dR, n_boot=args.n_boot, seed=args.seed + 1)
            mR, mdR = med(all_R), med(all_dR)
            win = sum(1 for r in all_R if r < 1) / len(all_R) if all_R else float("nan")

            J(f"**Unit-level (n = {len(all_R)})**: median `R_cross` = **{fmt(mR)}** "
              f"(95% CI [{fmt(ci_R[0])}, {fmt(ci_R[1])}]), IQR {fmt(iqr(all_R))}, "
              f"range {fmt(min(all_R))}-{fmt(max(all_R))}, cross better in {100 * win:.0f}%.")
            J("")
            J(f"**Unit-level effect size**: median `delta_rmse` = **{mdR:+.2f} min** "
              f"(95% CI [{ci_dR[0]:+.2f}, {ci_dR[1]:+.2f}]).")
            J("")

            # per-target stratification (protocol S8)
            J("| target | n units | median R_cross | 95% CI | win rate | median dRMSE (min) |")
            J("|---|---|---|---|---|---|")
            strat = {}
            for tgt in sorted({u["target"] for u in unit_rows}):
                rs = [u["R"] for u in unit_rows if u["target"] == tgt]
                ds_ = [u["dRMSE"] for u in unit_rows if u["target"] == tgt]
                c = boot_ci(rs, n_boot=args.n_boot, seed=args.seed + 7)
                strat[tgt] = {"n": len(rs), "median_R": med(rs), "ci_R": c,
                              "win_rate": sum(1 for r in rs if r < 1) / len(rs) if rs else None,
                              "median_dRMSE": med(ds_)}
                J(f"| {tgt} | {len(rs)} | {fmt(med(rs))} | [{fmt(c[0])}, {fmt(c[1])}] | "
                  f"{100 * strat[tgt]['win_rate']:.0f}% | {med(ds_):+.2f} |")
            J("")

            # fold-level robustness (protocol S3)
            fold_meds = []
            for f in folds:
                rs = [u["R"] for u in unit_rows if u["fold"] == f["name"]]
                if rs:
                    fold_meds.append((f["name"], med(rs), len(rs)))
            n_pos = sum(1 for _n, m, _c in fold_meds if m > 1.0)
            J("**Across folds (robustness only -- folds are NOT extra samples, S3)**: "
              + ", ".join(f"{n}={fmt(m)}" for n, m, _c in fold_meds)
              + f"; fold medians > 1 in **{n_pos}/{len(fold_meds)}**.")
            if len(fold_meds) >= 2:
                vals = [m for _n, m, _c in fold_meds]
                spread = 100.0 * (max(vals) - min(vals)) / statistics.fmean(vals)
                eff = abs(mR - 1.0) * 100.0
                J("")
                J(f"- |median R_cross - 1| = **{eff:.1f}%**  vs  between-fold spread "
                  f"= **{spread:.1f}%**")
                J(f"- fold-to-fold spread is a property of *which bearing is held out* "
                  f"(measured run-to-run noise is separate: `scripts/17_reproducibility_probe.py`).")
            J("")

            p_cell = wilcoxon([math.log(r) for r in cell_R if r > 0], 0.0)
            p_unit = wilcoxon([math.log(r) for r in all_R if r > 0], 0.0)
            J(f"**Statistics (explicitly downgraded, S7)**: "
              f"cell-level Wilcoxon on log R: "
              f"{'p = %.3g' % p_cell if p_cell is not None else 'n/a'} "
              f"(45 cells, **exploratory only -- not 45 independent samples**); "
              f"unit-level: "
              f"{'p = %.3g' % p_unit if p_unit is not None else 'n/a'} "
              f"(n = {len(all_R)}, low power, indicative only).")
            J("")
            J("> " + _EST_ADVICE)
            J("")

            report["metrics"].setdefault(metric, {})[model] = {
                "n_units": len(all_R),
                "units": unit_rows,
                "unit_level": {"median_R": mR, "ci_R": ci_R, "iqr_R": iqr(all_R),
                               "win_rate": win, "min_R": min(all_R), "max_R": max(all_R)},
                "effect_size": {"median_dRMSE_min": mdR, "ci_dRMSE_min": ci_dR},
                "by_target": strat,
                "fold_medians": fold_meds,
                "n_folds_median_above_one": n_pos,
                "p_cell_level_EXPLORATORY_ONLY": p_cell,
                "p_unit_level_indicative": p_unit,
                "n_cells_descriptive": len(cell_R),
                "median_R_cells_descriptive": med(cell_R),
            }

    J("---")
    J("")
    J("## Caveat (protocol S10)")
    J("")
    J("Even the 15 units are not strictly independent: across folds the TRAINING bearings")
    J("overlap, so the bearing-level bootstrap interval is a **lower bound** on the true")
    J("uncertainty. Report it as such.")
    J("")

    out_md = ROOT / args.out
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")

    out_json = ROOT / args.json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
