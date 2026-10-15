"""Stage 4 analysis: component contribution, label sensitivity, cross-condition transfer.

Frozen statistical structure (user, 2026-10-08; ``audit/STAGE4_PROTOCOL_FREEZE.md``
sections 3 and 5)::

    S4-T1  L1  component contribution dG (extractor / SE / bidirectional)
               + supplementary family comparison (cnn vs pure_bigru), reported apart
    S4-T2  L2  G_m(h), dG_architecture(h), observed crossover interval
    S4-T3  L3  architecture x cross-condition (dG_architecture / R_cross)

Metric definitions (inherited, NOT new -- freeze section 5)
-----------------------------------------------------------
``G_m = 1 - RMSE_m / RMSE_const_mean``  (``const_mean`` = the deployable
zero-learning baseline). ``const_mean`` is a trivial model, so its RMSE is
*identical across architectures* and cancels in every ``dG``: a ``dG`` is a
statement about absolute error relative to a shared floor.

Statistical unit (``audit/STATISTICAL_UNIT_PROTOCOL.md``)
--------------------------------------------------------
The unit is the TEST BEARING INSTANCE. For L1/L2 (single-condition folds, test
bearing 5 of each condition) that is 3 units (cond1/2/3); for L3 it is 15 units
(5 folds x 3 conditions = all 15 XJTU-SY bearings). Inside one unit the 3 source-task
variants are REPEATED MEASURES on the same bearing and are collapsed by median before
any statistic (protocol S2), so ``n_units`` reads 15, never 45. Evidence is the
DISTRIBUTION + effect size + a bearing-level bootstrap CI over the units.

Crossover (freeze section 3.3)
------------------------------
We report the **observed crossover interval** -- the pair of adjacent grid points
between which ``G`` changes sign -- and flag ``not identified`` when no sign
change is observed. We never report an interpolated critical value ``h*``.

Trains nothing. Pure re-analysis of ``artifacts/reports/s4/``.

Usage
-----
    python scripts/37_stage4_analysis.py
    python scripts/37_stage4_analysis.py --metric rmse --bootstrap 2000
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
import statistics
import sys
from typing import Dict, List, Optional, Sequence, Tuple

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

S4 = REPO / "artifacts" / "reports" / "s4"
OUT = REPO / "artifacts" / "reports" / "stage4_analysis"
TABLES = REPO / "paper" / "tables"

DISPLAY = {
    "se_resnet_bigru": "SE-ResNet-BiGRU",
    "resnet_bigru": "ResNet-BiGRU",
    "pure_bigru": "BiGRU",
    "se_resnet_gru": "SE-ResNet-GRU",
    "se_resnet_lstm": "SE-ResNet-LSTM",
    "resnet_lstm": "ResNet-BiLSTM",
    "cnn": "CNN",
}

#: the three component-contribution edges (frozen section 3.1)
COMPONENT_EDGES: List[Tuple[str, str, str]] = [
    ("pure_bigru", "resnet_bigru", "extractor"),
    ("resnet_bigru", "se_resnet_bigru", "SE"),
    ("se_resnet_gru", "se_resnet_bigru", "bidirectional"),
]
FAMILY_EDGE = ("cnn", "pure_bigru", "CNN -> BiGRU (family comparison)")

H_VALUES = [0.5, 0.6, 0.7, 0.8, 0.9]
L3_MODELS = ["se_resnet_bigru", "resnet_bigru", "pure_bigru"]


# ------------------------------------------------------------------ I/O utils
def read_results(d: pathlib.Path, metric: str = "rmse") -> Dict[Tuple[str, str], float]:
    """{(dataset, model): metric} from ``results_<metric>.csv`` under ``d``."""
    out: Dict[Tuple[str, str], float] = {}
    if not d.exists():
        return out
    for p in sorted(d.glob(f"*/results_{metric}.csv")) + sorted(d.glob(f"results_{metric}.csv")):
        try:
            with p.open("r", encoding="utf-8-sig", newline="") as fh:
                for r in csv.DictReader(fh):
                    key = (r.get("dataset", ""), r.get("model", ""))
                    val = r.get(metric, "")
                    if val not in ("", None):
                        out[key] = float(val)
        except (OSError, ValueError):
            continue
    return out


def rmse_by_condition(d: pathlib.Path, model: str, metric: str = "rmse") -> Dict[str, float]:
    """{condition: metric} for one model under one report dir."""
    out: Dict[str, float] = {}
    p = d / model / f"results_{metric}.csv"
    if not p.exists():
        p = d / f"{model}" / "results_corrected.csv"
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            cond = r.get("protocol") or r.get("condition") or ""
            # the single-condition files store the condition in ``dataset`` name
            ds = r.get("dataset", "")
            for tag in ("cond1", "cond2", "cond3"):
                if tag in ds:
                    cond = tag
            val = r.get(metric, "")
            if cond in ("cond1", "cond2", "cond3") and val not in ("", None):
                out[cond] = float(val)
    return out


def _load_cond_map(csv_path: pathlib.Path) -> Dict[str, float]:
    """{cond1|cond2|cond3: rmse} from a results-related csv.

    The XJTU-SY report files carry the condition in the ``dataset`` column
    (``cond1`` / ``cond2`` / ``cond3``). Fall back to row order only when that
    column is absent, so a renamed column cannot silently permute the mapping.
    """
    out: Dict[str, float] = {}
    if not csv_path.exists():
        return out
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    tags = ["cond1", "cond2", "cond3"]
    for i, r in enumerate(rows):
        tag = None
        for cand in tags:
            if str(r.get("dataset", "")).startswith(cand):
                tag = cand
                break
        if tag is None and i < len(tags):
            tag = tags[i]
        if tag is None:
            continue
        try:
            out[tag] = float(r["rmse"])
        except (KeyError, ValueError):
            pass
    return out


# ----------------------------------------------------------------- bootstrap
def bootstrap_ci(values: Sequence[float], n: int = 2000, alpha: float = 0.05,
                 seed: int = 20261008) -> Tuple[float, float, float]:
    """Percentile bootstrap CI of the median, resampling the observed units."""
    vals = [v for v in values if v == v]  # drop NaN
    if not vals:
        return float("nan"), float("nan"), float("nan")
    med = statistics.median(vals)
    if len(vals) == 1:
        return med, med, med
    rng = random.Random(seed)
    boot = []
    for _ in range(n):
        s = [vals[rng.randrange(len(vals))] for _ in vals]
        boot.append(statistics.median(s))
    boot.sort()
    lo = boot[int(alpha / 2 * n)]
    hi = boot[min(n - 1, int((1 - alpha / 2) * n))]
    return med, lo, hi


# ------------------------------------------------------------------- L1 / S4-T1
def _const_mean_per_condition(root: pathlib.Path, subdirs: Sequence[str],
                              model: str = "constant_mean") -> Dict[str, float]:
    """const_mean RMSE per condition, read from the first report subdir that has it.

    ``subdirs`` are report-directory names (e.g. ``["l1_linear", "l2_h0.5"]``),
    not model names -- the floor lives at ``<root>/<subdir>/<model>/``.
    """
    for sub in subdirs:
        m = _load_cond_map(root / sub / model / "results_corrected.csv")
        if m:
            return m
    return {}


def l1_component_table(root: pathlib.Path, metric: str = "rmse") -> dict:
    """S4-T1 -- component dG on the shared const_mean floor, per condition."""
    l1 = root / "l1_linear"
    floor = _const_mean_per_condition(root, ["l1_linear"])
    models = [e[0] for e in COMPONENT_EDGES] + [e[1] for e in COMPONENT_EDGES] + [FAMILY_EDGE[0]]
    models = sorted(set(models), key=lambda m: [x for e in COMPONENT_EDGES for x in e[:2]].index(m)
                    if m in [x for e in COMPONENT_EDGES for x in e[:2]] else 99)
    per_model = {m: _load_cond_map(l1 / m / "results_corrected.csv") for m in models}

    def G(model: str, cond: str) -> float:
        v, f = per_model.get(model, {}).get(cond), floor.get(cond)
        if v is None or not f:
            return float("nan")
        return 1.0 - v / f

    out = {"floor_by_condition": floor, "G_by_model": {}, "component_dG": [], "family_comparison": []}
    for m in models:
        out["G_by_model"][m] = {c: G(m, c) for c in ("cond1", "cond2", "cond3")}
        out["G_by_model"][m]["mean"] = statistics.mean(
            [G(m, c) for c in ("cond1", "cond2", "cond3") if G(m, c) == G(m, c)] or [float("nan")])

    for a, b, label in COMPONENT_EDGES:
        d = {c: G(b, c) - G(a, c) for c in ("cond1", "cond2", "cond3")}
        vals = [v for v in d.values() if v == v]
        med, lo, hi = bootstrap_ci(vals)
        out["component_dG"].append({
            "edge": f"{DISPLAY.get(a, a)} -> {DISPLAY.get(b, b)}",
            "factor": label, "a": a, "b": b,
            "dG_by_condition": d, "dG_median": med, "ci": [lo, hi],
            "sign_consistent": all(v >= 0 for v in vals) or all(v <= 0 for v in vals),
        })

    a, b, label = FAMILY_EDGE
    d = {c: G(b, c) - G(a, c) for c in ("cond1", "cond2", "cond3")}
    vals = [v for v in d.values() if v == v]
    med, lo, hi = bootstrap_ci(vals)
    out["family_comparison"].append({
        "edge": "CNN -> BiGRU", "note": label, "a": a, "b": b,
        "dG_by_condition": d, "dG_median": med, "ci": [lo, hi],
    })
    return out


# ------------------------------------------------------------------- L2 / S4-T2
def l2_surface(root: pathlib.Path) -> dict:
    """S4-T2 -- G_m(h), dG_architecture(h), observed crossover interval."""
    # h -> model -> {condition: rmse}
    grid: Dict[str, Dict[str, Dict[str, float]]] = {}
    points = [("linear", "l1_linear")] + [(f"h={h}", f"l2_h{h}") for h in H_VALUES]
    models = ["se_resnet_bigru", "resnet_bigru", "pure_bigru", "se_resnet_gru"]
    for tag, sub in points:
        d = root / sub
        floor = _const_mean_per_condition(root, [sub])
        grid[tag] = {"_floor": floor}
        for m in models:
            grid[tag][m] = _load_cond_map(d / m / "results_corrected.csv")

    def G(tag: str, model: str) -> float:
        fl = grid[tag]["_floor"]
        cm = grid[tag].get(model, {})
        vals = [1.0 - cm[c] / fl[c] for c in ("cond1", "cond2", "cond3") if c in cm and fl.get(c)]
        return statistics.mean(vals) if vals else float("nan")

    Gs: Dict[str, Dict[str, float]] = {m: {} for m in models}
    for tag, _ in points:
        for m in models:
            Gs[m][tag] = G(tag, m)

    def crossover(series: List[float], tags: List[str]) -> dict:
        """Observed sign change between ADJACENT grid points; never interpolated."""
        signs = []
        for v in series:
            signs.append(None if v != v else (1 if v > 0 else (-1 if v < 0 else 0)))
        for i in range(len(signs) - 1):
            a, b = signs[i], signs[i + 1]
            if a is None or b is None:
                continue
            if a != b and a != 0 and b != 0:
                return {"interval": [tags[i], tags[i + 1]], "lo": series[i], "hi": series[i + 1],
                        "identified": True}
        return {"interval": None, "identified": False, "note": "not identified"}

    tags = [t for t, _ in points]
    out = {"grid_tags": tags, "G_surface": {}, "dG_surface": [], "crossover": {}}
    for m in models:
        out["G_surface"][m] = {t: Gs[m][t] for t in tags}

    dG_edges = [("resnet_bigru", "se_resnet_bigru", "SE"),
                ("pure_bigru", "resnet_bigru", "extractor"),
                ("se_resnet_gru", "se_resnet_bigru", "bidirectional")]
    for a, b, label in dG_edges:
        series = [Gs[b][t] - Gs[a][t] for t in tags]
        out["dG_surface"].append({
            "edge": f"{DISPLAY.get(a, a)} -> {DISPLAY.get(b, b)}", "factor": label,
            "by_label": {t: series[i] for i, t in enumerate(tags)},
        })

    for m in models:
        out["crossover"][m] = crossover([Gs[m][t] for t in tags], tags)
    return out


# ------------------------------------------------------------------- L3 / S4-T3
def _parse_task(task: str) -> Optional[Tuple[List[str], str]]:
    """``"1+2->3"`` -> ``(["1", "2"], "3")``; ``None`` when not a task id."""
    if "->" not in task:
        return None
    src, tgt = task.split("->")
    return src.split("+"), tgt


def _cond_of_single(ds: str) -> Optional[str]:
    """``"cond2"`` -> ``"2"``; ``None`` when not a single-condition row."""
    return ds[4:] if ds.startswith("cond") else None


def _read_arm(arm_dir: pathlib.Path, model: str, metric: str = "rmse") -> Dict[str, float]:
    """{(dataset): metric} for one model in one arm directory.

    Reads ``results_corrected.csv`` when present (the ``_b8`` vintage), else the
    first ``results_<metric>.csv`` found. Only rows for ``model`` are kept.
    """
    out: Dict[str, float] = {}
    p = arm_dir / "results_corrected.csv"
    if not p.exists():
        p = arm_dir / f"results_{metric}.csv"
    if not p.exists():
        cands = sorted(arm_dir.rglob(f"results_{metric}.csv"))
        if not cands:
            return out
        p = cands[0]
    with p.open("r", encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("model") != model:
                continue
            ds = r.get("dataset", "")
            val = r.get(metric, "")
            if ds and val not in ("", None):
                try:
                    out[ds] = float(val)
                except ValueError:
                    continue
    return out


def _paired_units(single: Dict[str, float], cross: Dict[str, float]
                  ) -> Dict[str, Tuple[float, float, float]]:
    """Collapse the 3 source-task variants into the frozen ``(condition, bearing)`` unit.

    The unit is the TEST BEARING INSTANCE ``(fold, target condition)``
    (``audit/STATISTICAL_UNIT_PROTOCOL.md``). Inside one unit the up-to-3 source
    tasks are REPEATED MEASURES on the same bearing, so they are pre-aggregated by
    the median (protocol S2), exactly as ``scripts/23_stat_unit_rcross.py`` does.
    This is what makes ``n_units == 15`` for the five matched folds.

    Returns ``{target_condition: (R_cross, rmse_single, rmse_cross)}``.
    """
    per_unit: Dict[str, List[Tuple[float, float, float]]] = {}
    for ds, inc in single.items():
        tgt = _cond_of_single(ds)
        if tgt is None or not (inc and inc > 0):
            continue
        for task, cx in cross.items():
            pt = _parse_task(task)
            if not pt or pt[1] != tgt:
                continue
            per_unit.setdefault(tgt, []).append((cx / inc, inc, cx))
    out: Dict[str, Tuple[float, float, float]] = {}
    for tgt, recs in per_unit.items():
        out[tgt] = (statistics.median(r[0] for r in recs),
                    statistics.median(r[1] for r in recs),
                    statistics.median(r[2] for r in recs))
    return out


def l3_table(root: pathlib.Path) -> dict:
    """S4-T3 -- architecture x cross-condition: paired R_cross and dG.

    The unit is the ``(fold, target condition)`` bearing: 5 folds x 3 conditions =
    **15 units** = all 15 XJTU-SY bearings. The 3 source-task variants inside a unit
    are repeated measures and are collapsed by median before any statistic -- so
    ``n_units`` must read 15, never 45 (freeze red line 30, protocol S2).

    Both arms share a byte-identical test bearing, so ``R_cross = RMSE_cross /
    RMSE_single`` needs no recalibration; it is the uncorrected caliber only
    (freeze red line 22).
    """
    rows = []
    for m in L3_MODELS:
        units_rcross: List[float] = []
        units_drmse: List[float] = []
        for k in range(1, 6):
            base = root / f"l3_fold{k}"
            if not base.exists():
                continue
            s = _read_arm(base / "single", m)
            c = _read_arm(base / "cross", m)
            for _tgt, (R, inc, cx) in _paired_units(s, c).items():
                units_rcross.append(R)
                units_drmse.append(cx - inc)
        med, lo, hi = bootstrap_ci(units_rcross)
        med_d, lo_d, hi_d = bootstrap_ci(units_drmse)
        rows.append({
            "model": m, "display": DISPLAY.get(m, m),
            "n_units": len(units_rcross),
            "R_cross_median": med, "R_cross_ci": [lo, hi],
            "dRMSE_median": med_d, "dRMSE_ci": [lo_d, hi_d],
            "cross_better_frac": (sum(1 for v in units_rcross if v < 1) / len(units_rcross)
                                  if units_rcross else float("nan")),
        })
    return {"rows": rows}


# --------------------------------------------------------------------- report
def write_tables(l1: dict, l2: dict, l3: dict, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: List[str] = ["# Stage 4 tables (auto-generated)", ""]

    # S4-T1
    lines.append("### S4-T1 — L1 component contribution (paired dG vs BiGRU's dG chain)")
    lines.append("")
    lines.append("| Edge | Factor | ΔG cond1 | ΔG cond2 | ΔG cond3 | ΔG median | 95% CI |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for e in l1["component_dG"]:
        d = e["dG_by_condition"]
        lines.append(f"| {e['edge']} | {e['factor']} | {d['cond1']:+.3f} | {d['cond2']:+.3f} | "
                     f"{d['cond3']:+.3f} | **{e['dG_median']:+.3f}** | "
                     f"[{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}] |")
    for e in l1["family_comparison"]:
        d = e["dG_by_condition"]
        lines.append(f"| *{e['edge']}* (supplementary) | *family comparison* | {d['cond1']:+.3f} | "
                     f"{d['cond2']:+.3f} | {d['cond3']:+.3f} | "
                     f"*{e['dG_median']:+.3f}* | *[{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]* |")
    lines.append("")

    # S4-T2
    tags = l2["grid_tags"]
    lines.append("### S4-T2 — L2 label structure × architecture: G_m(h)")
    lines.append("")
    lines.append("| Model | " + " | ".join(tags) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in tags) + " |")
    for m, series in l2["G_surface"].items():
        cells = ["—" if v != v else f"{v:+.3f}" for v in (series[t] for t in tags)]
        lines.append(f"| {DISPLAY.get(m, m)} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("| ΔG edge | Factor | " + " | ".join(tags) + " |")
    lines.append("| --- | --- | " + " | ".join("---" for _ in tags) + " |")
    for e in l2["dG_surface"]:
        cells = [f"{e['by_label'][t]:+.3f}" for t in tags]
        lines.append(f"| {e['edge']} | {e['factor']} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("**Observed crossover interval** (no interpolated h* is reported):")
    lines.append("")
    lines.append("| Model | crossover |")
    lines.append("| --- | --- |")
    for m, c in l2["crossover"].items():
        txt = (f"{c['interval'][0]} → {c['interval'][1]}" if c.get("identified")
               else "*not identified*")
        lines.append(f"| {DISPLAY.get(m, m)} | {txt} |")
    lines.append("")

    # S4-T3
    lines.append("### S4-T3 — L3 architecture × cross-condition")
    lines.append("")
    lines.append("Unit = `(fold, target condition)` bearing; 5 folds × 3 conditions = **15 units**")
    lines.append("(the 3 source-task variants inside a unit are repeated measures, collapsed by median).")
    lines.append("")
    lines.append("`R_cross` and `ΔRMSE` (min) are reported **together**, always with `n` "
                 "(protocol S5); `R_cross>1` alone is not evidence.")
    lines.append("")
    lines.append("| Model | n units | R_cross median | 95% CI | ΔRMSE median (min) | 95% CI | cross better |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in l3["rows"]:
        lines.append(f"| {r['display']} | {r['n_units']} | {r['R_cross_median']:.3f} | "
                     f"[{r['R_cross_ci'][0]:.3f}, {r['R_cross_ci'][1]:.3f}] | "
                     f"{r['dRMSE_median']:+.2f} | "
                     f"[{r['dRMSE_ci'][0]:+.2f}, {r['dRMSE_ci'][1]:+.2f}] | "
                     f"{r['cross_better_frac'] * 100:.0f}% |")
    lines.append("")

    (out_dir / "stage4_tables.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(S4))
    ap.add_argument("--metric", default="rmse")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    if not root.exists():
        print(f"[error] {root} not found -- run scripts/36_run_stage4.py first")
        return 2

    l1 = l1_component_table(root, args.metric)
    l2 = l2_surface(root)
    l3 = l3_table(root)

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"l1": l1, "l2": l2, "l3": l3, "metric": args.metric}
    (out_dir / "stage4_analysis.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_tables(l1, l2, l3, TABLES)

    print(f"[s4-analysis] wrote {out_dir / 'stage4_analysis.json'}")
    print(f"[s4-analysis] wrote {TABLES / 'stage4_tables.md'}")
    print()
    print("== S4-T1 component dG ==")
    for e in l1["component_dG"]:
        print(f"  {e['factor']:<14} {e['edge']:<34} median dG={e['dG_median']:+.3f} "
              f"CI=[{e['ci'][0]:+.3f},{e['ci'][1]:+.3f}]")
    print("== S4-T2 crossover ==")
    for m, c in l2["crossover"].items():
        txt = f"{c['interval'][0]} -> {c['interval'][1]}" if c.get("identified") else "not identified"
        print(f"  {DISPLAY.get(m, m):<18} {txt}")
    print("== S4-T3 architecture x cross-condition (unit = fold x condition) ==")
    for r in l3["rows"]:
        print(f"  {r['display']:<18} n={r['n_units']:<3} R_cross={r['R_cross_median']:.3f} "
              f"CI=[{r['R_cross_ci'][0]:.3f},{r['R_cross_ci'][1]:.3f}] "
              f"dRMSE={r['dRMSE_median']:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
