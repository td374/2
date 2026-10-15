"""Matched in-condition vs cross-condition comparison.

Motivation
----------
The S2L (in-condition) and S3 (cross-condition) runs do not test on the same
population:

* ``splits.single``  train {1,2,3} / val {4} / test {5}          -> ONE test bearing
* ``splits.cross``   source {1,2,3,4} / source_val {5} / target_test {1..5}
                                                                 -> ALL FIVE test bearings

The cross aggregate is therefore dominated by long-life bearings (condition 3
contains bearings of 371-2538 min next to the 114 min one), while the
in-condition number is measured on bearing ``_5`` alone. Aggregated cross RMSE
and in-condition RMSE are consequently not comparable.

``results_full.json`` keeps per-bearing metrics for every (task, model). This
script re-aggregates BOTH protocols on the **matched** test set (bearing ``_5``
of each condition) so the only remaining difference is the training data
(own condition vs other conditions).

Outputs
-------
* ``artifacts/reports/matched_comparison.json``
* ``docs/matched_cross_vs_incondition.md``

Pure re-analysis: no training, reads only files already on disk.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORTS = ROOT / "artifacts" / "reports"

SINGLE_DIR = REPORTS / "single_linear"
CROSS_ARMS = {
    "UDA on": REPORTS / "cross_uda_linear",
    "source only": REPORTS / "cross_source_only_linear",
}

MODELS = ["se_resnet_bigru", "cnn", "lstm", "gru"]
# trivial baselines are only present in the cross arm
BASELINES = ["constant_one", "constant_mean", "ridge"]

COND_LIFE_NOTE = "condition 3 mixes 114 min bearings with 2496-2538 min bearings"


# --------------------------------------------------------------------- helpers
def load_full(path: pathlib.Path):
    """Return list of records from a results_full.json, or [] if absent."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        print(f"  ! cannot read {path}: {exc}", file=sys.stderr)
        return []
    if isinstance(data, dict):
        data = list(data.values())
    return data


def per_bearing_map(records):
    """{dataset: {model: {bearing_id: {'rmse':.., 'n':..}}}}"""
    out: dict = {}
    for r in records:
        ds = r.get("dataset")
        model = r.get("model")
        pb = r.get("per_bearing") or []
        out.setdefault(ds, {}).setdefault(model, {})
        for b in pb:
            bid = int(b.get("bearing_id"))
            out[ds][model][bid] = {
                "rmse": float(b.get("rmse", float("nan"))),
                "mae": float(b.get("mae", float("nan"))),
                "n": int(b.get("n_windows", 0)),
            }
    return out


def pooled_rmse(bearing_ms_rmse):
    """RMSE pooled over several bearing groups from their per-group RMSE and n."""
    num = 0.0
    den = 0
    for rmse, n in bearing_ms_rmse:
        if n <= 0 or not math.isfinite(rmse):
            continue
        num += n * rmse * rmse
        den += n
    return (math.sqrt(num / den) if den else float("nan")), den


def parse_task(task: str):
    """'1+2->3' -> (['1','2'], '3')"""
    src, tgt = task.split("->")
    return src.split("+"), tgt


# ----------------------------------------------------------------- load inputs
print("loading single-condition records ...")
single_by_cond = {}
for m in MODELS:
    for r in load_full(SINGLE_DIR / m / "results_full.json"):
        ds = r.get("dataset")  # e.g. 'cond1'
        if not ds or not ds.startswith("cond"):
            continue
        cond = ds.replace("cond", "")
        single_by_cond.setdefault(cond, {})[r.get("model")] = r
for c in sorted(single_by_cond):
    print(f"  single cond{c}: models={sorted(single_by_cond[c])}")

cross = {}
for arm, d in CROSS_ARMS.items():
    recs = load_full(d / "results_full.json")
    cross[arm] = per_bearing_map(recs)
    print(f"  {arm}: {len(recs)} records, tasks={len(cross[arm])}")

# tasks grouped by target condition
tasks_by_target: dict = {}
for arm in cross:
    for task in cross[arm]:
        _, tgt = parse_task(task)
        tasks_by_target.setdefault(tgt, set()).add(task)
for t in sorted(tasks_by_target):
    print(f"  target cond{t}: {sorted(tasks_by_target[t])}")


# ------------------------------------------ analysis 1: matched (bearing 5 only)
def matched_table(arm: str):
    """Per target condition: in-condition RMSE vs cross RMSE on bearing _5."""
    rows = []
    for tgt in ["1", "2", "3"]:
        tasks = sorted(tasks_by_target.get(tgt, []))
        row = {"target": tgt, "tasks": tasks, "incond": {}, "cross": {}}

        # in-condition: the single run for this condition tests bearing _5 only
        for m in MODELS:
            rec = single_by_cond.get(tgt, {}).get(m)
            if not rec:
                continue
            pb = {int(b["bearing_id"]): b for b in rec.get("per_bearing", [])}
            if 5 in pb:
                row["incond"][m] = {
                    "rmse": float(pb[5]["rmse"]),
                    "n": int(pb[5]["n_windows"]),
                }

        # cross: bearing _5 of the target condition, per task
        for task in tasks:
            srcs, _ = parse_task(task)
            for m in MODELS + BASELINES:
                bmap = cross[arm].get(task, {}).get(m, {})
                if 5 in bmap:
                    row["cross"].setdefault(m, {})[task] = {
                        "sources": srcs,
                        "rmse": bmap[5]["rmse"],
                        "n": bmap[5]["n"],
                    }
        rows.append(row)
    return rows


# --------------------------- analysis 1b: the cross/single gain ratio R_cross
def cross_gain_table(arm: str):
    """R_cross = RMSE_cross / RMSE_single on the matched test bearing.

    R < 1 -> cross-condition helps.  R > 1 -> cross-condition hurts.
    Only the *best* source variant per (target, model) is used, which is the
    most conservative choice against the "cross is worse" claim.
    """
    rows = []
    for tgt in ["1", "2", "3"]:
        for m in MODELS:
            inc = None
            rec = single_by_cond.get(tgt, {}).get(m)
            if rec:
                pb = {int(b["bearing_id"]): b for b in rec.get("per_bearing", [])}
                if 5 in pb:
                    inc = float(pb[5]["rmse"])
            best_task, best = None, float("inf")
            for task in tasks_by_target.get(tgt, []):
                bmap = cross[arm].get(task, {}).get(m, {})
                if 5 in bmap and bmap[5]["rmse"] < best:
                    best_task, best = task, bmap[5]["rmse"]
            if inc is None or best_task is None:
                continue
            ratio = best / inc
            rows.append(
                {
                    "target": tgt,
                    "model": m,
                    "rmse_single": inc,
                    "rmse_cross": best,
                    "cross_task": best_task,
                    "ratio": ratio,
                    "delta_pct": 100.0 * (ratio - 1.0),
                }
            )
    return rows


# --------------------------------- analysis 2: multi-source vs single-source
def source_count_table(arm: str):
    """Fully paired: same target condition, same 5-bearing test population."""
    rows = []
    for tgt in ["1", "2", "3"]:
        for m in MODELS:
            single_src = {}
            dual = None
            for task in tasks_by_target.get(tgt, []):
                srcs, _ = parse_task(task)
                bmap = cross[arm].get(task, {}).get(m, {})
                if not bmap:
                    continue
                # pooled over the whole target test population
                pooled, n = pooled_rmse(
                    [(b["rmse"], b["n"]) for b in bmap.values()]
                )
                if len(srcs) == 1:
                    single_src[srcs[0]] = pooled
                else:
                    dual = pooled
            if not single_src or dual is None:
                continue
            best = min(single_src.values())
            worst = max(single_src.values())
            rows.append(
                {
                    "target": tgt,
                    "model": m,
                    "single": single_src,
                    "dual": dual,
                    "best_single": best,
                    "worst_single": worst,
                    "between": worst >= dual >= best,
                    "dual_vs_best_pct": 100.0 * (dual - best) / best,
                }
            )
    return rows


# ------------------------------- analysis 3: how much of the aggregate is scale
def population_table(arm: str):
    rows = []
    for tgt in ["1", "2", "3"]:
        for task in sorted(tasks_by_target.get(tgt, [])):
            for m in MODELS:
                bmap = cross[arm].get(task, {}).get(m, {})
                if not bmap:
                    continue
                agg, n_all = pooled_rmse([(b["rmse"], b["n"]) for b in bmap.values()])
                only5 = bmap.get(5, {})
                if not only5:
                    continue
                rows.append(
                    {
                        "task": task,
                        "model": m,
                        "agg_all_bearings": agg,
                        "n_all": n_all,
                        "bearing5_only": only5["rmse"],
                        "n_bearing5": only5["n"],
                        "share_of_bearing5_pct": 100.0 * only5["n"] / n_all if n_all else float("nan"),
                    }
                )
    return rows


results = {}
for arm in cross:
    results[arm] = {
        "matched_bearing5": matched_table(arm),
        "cross_gain": cross_gain_table(arm),
        "source_count": source_count_table(arm),
        "population": population_table(arm),
    }

out_json = ROOT / "artifacts" / "reports" / "matched_comparison.json"
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nwrote {out_json}")


# ------------------------------------------------------------------- markdown
def fmt(x, nd=2):
    if x is None:
        return "-"
    try:
        v = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(v):
        return "-"
    return f"{v:.{nd}f}"


L = []
L.append("# Matched in-condition vs cross-condition comparison")
L.append("")
L.append("Pure re-analysis of `results_full.json` (no training). Reproduce with")
L.append("`python scripts/18_matched_incondition_vs_cross.py`.")
L.append("")
L.append("## 0. Why the headline numbers were not comparable")
L.append("")
L.append("| protocol | training | validation | **test population** |")
L.append("|---|---|---|---|")
L.append("| `single` (in-condition) | target cond bearings 1,2,3 | bearing 4 | **bearing 5 only** |")
L.append("| `cross` | source cond bearings 1,2,3,4 | source bearing 5 | **all five bearings** |")
L.append("")
L.append(f"Condition 3 mixes one 114 min bearing with 2496-2538 min bearings, so the")
L.append("cross aggregate is dominated by long-life units while the in-condition number")
L.append("is measured on the short-life bearing 5. The two are **not** the same population.")
L.append("")

for arm in cross:
    r = results[arm]
    L.append(f"## Arm: **{arm}**")
    L.append("")
    L.append("### 1. Matched test set (bearing `_5` only)")
    L.append("")
    L.append("| target cond | model | in-condition RMSE | cross RMSE (best source) | cross/in-cond |")
    L.append("|---|---|---|---|---|")
    for row in r["matched_bearing5"]:
        for m in MODELS:
            inc = row["incond"].get(m)
            cx = row["cross"].get(m)
            if not inc or not cx:
                continue
            best_task, best = None, float("inf")
            for task, d in cx.items():
                if d["rmse"] < best:
                    best_task, best = task, d["rmse"]
            ratio = best / inc["rmse"] if inc["rmse"] else float("nan")
            L.append(
                f"| {row['target']} | {m} | {fmt(inc['rmse'])} | "
                f"{fmt(best)} ({best_task}) | {fmt(ratio, 3)}× |"
            )
    L.append("")
    L.append("### 2. Cross / single gain ratio `R_cross` (matched test bearing)")
    L.append("")
    L.append("`R_cross = RMSE_cross / RMSE_single` on the same test bearing `_5`. "
             "`R < 1` means cross-condition helps; `R > 1` means it hurts. "
             "Only the best source variant per (target, model) is used.")
    L.append("")
    L.append("| target cond | model | single-cond RMSE | cross RMSE (best source) | R_cross | delta |")
    L.append("|---|---|---|---|---|---|")
    for row in r["cross_gain"]:
        L.append(
            f"| {row['target']} | {row['model']} | {fmt(row['rmse_single'])} | "
            f"{fmt(row['rmse_cross'])} ({row['cross_task']}) | "
            f"**{fmt(row['ratio'], 3)}** | {row['delta_pct']:+.1f}% |"
        )
    ratios = sorted(x["ratio"] for x in r["cross_gain"])
    if ratios:
        n = len(ratios)
        med = (ratios[n // 2] if n % 2 else 0.5 * (ratios[n // 2 - 1] + ratios[n // 2]))
        n_better = sum(1 for v in ratios if v < 1.0)
        L.append("")
        L.append(
            f"**Summary**: median `R_cross` = **{med:.3f}**; "
            f"cross-condition better in **{n_better}/{n}** cells; "
            f"range {min(ratios):.3f}-{max(ratios):.3f}."
        )
        L.append("")
        L.append("> ⚠ Per rule **A7** the run-to-run spread reaches ~31% on a single task, so")
        L.append("> individual cells here are not claimable. Only the *dispersion* and the")
        L.append("> *absence of a stable direction* are robust statements.")
    L.append("")
    L.append("### 3. Multi-source vs single-source (fully paired, same 5-bearing test set)")
    L.append("")
    L.append("| target cond | model | single sources | dual | best single | dual / best single | dual between? |")
    L.append("|---|---|---|---|---|---|---|")
    for row in r["source_count"]:
        ss = " / ".join(f"{k}->{row['target']}: {fmt(v)}" for k, v in sorted(row["single"].items()))
        L.append(
            f"| {row['target']} | {row['model']} | {ss} | {fmt(row['dual'])} | "
            f"{fmt(row['best_single'])} | {fmt(row['dual_vs_best_pct'], 1)}% | "
            f"{'yes' if row['between'] else 'NO'} |"
        )
    L.append("")
    L.append("### 4. How much of the cross aggregate is test-population composition")
    L.append("")
    L.append("| task | model | all bearings (pooled) | bearing 5 only | windows on bearing 5 |")
    L.append("|---|---|---|---|---|")
    for row in r["population"]:
        L.append(
            f"| {row['task']} | {row['model']} | {fmt(row['agg_all_bearings'])} | "
            f"{fmt(row['bearing5_only'])} | {row['share_of_bearing5_pct']:.1f}% |"
        )
    L.append("")

out_md = ROOT / "docs" / "matched_cross_vs_incondition.md"
out_md.write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"wrote {out_md}")
