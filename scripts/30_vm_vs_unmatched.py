"""Stage 3.2 verdict: does equalising the training volume change R_cross?

The unmatched matched-fold pilot (``artifacts/reports/matched_fold{1..5}``) fixes
the test object but lets the training volume float: ``scripts/22`` measured
cross/single ``n_train`` ratios of **0.07x-18.35x**, always MORE for target 1 and
always LESS for target 3. The volume-matched arm
(``artifacts/reports/vm_fold{1..5}``, built by ``scripts/28``) caps every arm of a
unit to the same ``K`` windows.

This script puts the two side by side at the frozen statistical unit
(``(fold, target)`` == ``(condition, bearing)``, n = 15) and reports:

1. unit-level ``R_cross`` and ``Delta RMSE`` (minutes) for both designs,
2. the **paired** difference ``R_matched - R_unmatched`` per unit, with a
   bearing-level percentile bootstrap CI,
3. the stratification by target condition -- the decisive read-out, because the
   volume confound's sign flips with the target: if target 3's excess R_cross
   (median 1.181 in the unmatched arm, where its cross arms saw 0.07-0.27x the
   data) collapses toward 1.0, the excess was a *volume* effect, not domain shift,
4. the realised n_train ratio per unit in each design (must be 1.00 when matched).

Discipline: ``R_cross`` is reported with the **uncorrected** caliber only -- the
median-ratio correction is applied to the cross arm alone (A9), so it cannot be
used across protocols. p-values are downgraded per the statistical-unit protocol.

Pure re-analysis. Trains nothing.

Usage
-----
    python scripts/30_vm_vs_unmatched.py
    python scripts/30_vm_vs_unmatched.py --model se_resnet_bigru
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import pathlib
import random
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

UNMATCHED = [f"artifacts/reports/matched_fold{k}_b8" for k in range(1, 6)]
MATCHED = [f"artifacts/reports/vm_fold{k}" for k in range(1, 6)]
DATA_UNMATCHED = [f"artifacts/data_fold{k}" for k in range(1, 6)]
DATA_MATCHED = [f"artifacts/data_vm_fold{k}" for k in range(1, 6)]


# ------------------------------------------------------------------- helpers
def load_results(d: pathlib.Path, metric: str = "uncorrected"):
    """{(dataset, model): rmse} from results_<metric>.csv under ``d``."""
    out = {}
    if not d.exists():
        return out
    for path in sorted(d.glob(f"results_{metric}.csv")):
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
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if len(vals) < 4:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(vals)
    stats = sorted(med([vals[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    return (stats[int(alpha / 2 * n_boot)], stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))])


def wilcoxon(xs, mu=0.0):
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
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "-"
    return f"{v:.{nd}f}" if math.isfinite(v) else "-"


def n_train_of(path: pathlib.Path) -> int:
    with np.load(path, allow_pickle=True) as z:
        return int(len(z["train/ratio"]))


def derive_data_dir(report_dir: str, from_tag: str, to_tag: str) -> str:
    """``artifacts/reports/matched_fold1_b8`` -> ``artifacts/data_fold1``.

    The ``_b8`` suffix marks a report re-evaluated with the current (post-B8)
    code; it is a report-vintage tag only and must not leak into the data dir.
    """
    name = pathlib.Path(report_dir).name.replace("_b8", "")
    return "artifacts/" + name.replace(from_tag, to_tag)


def earlystop_stats(report_dir: str, model: str) -> dict:
    """Read ``best_epoch`` / ``epochs_run`` off the checkpoints (no retraining).

    The volume-matched design equalises the TRAIN split but leaves the val split
    as it was, and the val splits are very unequal (11 windows for fold3/target2's
    in-condition arm vs 91 for its cross arm). Early stopping consumes the val
    split, so this records whether the two arms actually stop at different points
    -- i.e. how large the residual asymmetry really is.
    """
    import torch

    out: dict = {}
    for arm in ("single", "cross"):
        ck = ROOT / report_dir / arm / "checkpoints"
        if not ck.is_dir():
            continue
        be, er, bv = [], [], []
        for p in sorted(ck.glob(f"*__{model}__seed*.pt")):
            try:
                payload = torch.load(p, map_location="cpu", weights_only=False)
            except Exception:
                continue
            if "best_epoch" in payload:
                be.append(int(payload["best_epoch"]))
            if "epochs_run" in payload:
                er.append(int(payload["epochs_run"]))
            if "best_val_loss" in payload:
                bv.append(float(payload["best_val_loss"]))
        if be:
            out[arm] = {"n": len(be), "median_best_epoch": med(be),
                        "median_epochs_run": med(er), "min_best_epoch": min(be),
                        "max_best_epoch": max(be)}
    return out


# ------------------------------------------------------------------- table
def fold_id(report_dir: str) -> str:
    """Canonical fold key.

    ``artifacts/reports/matched_fold3_b8`` and ``artifacts/reports/vm_fold3`` must
    map to the SAME unit key, otherwise the two designs never overlap -- the
    directory basename is a vintage tag, not part of the unit identity.
    """
    m = re.search(r"(\d+)", pathlib.Path(report_dir).name)
    return m.group(1) if m else pathlib.Path(report_dir).name


def build_units(report_dirs, data_dirs, model, metric="uncorrected"):
    """{(fold, target): {'R':…, 'dRMSE':…, 'ratio':…, 'K':…}} unit-level medians."""
    units: dict = {}
    for rd, dd in zip(report_dirs, data_dirs):
        base = ROOT / rd
        fold = fold_id(rd)
        s = load_results(base / "single", metric)
        c = load_results(base / "cross", metric)
        data_dir = ROOT / dd
        single_n = {}
        cross_n = {}
        for p in sorted(data_dir.glob("single_cond*.npz")):
            single_n[p.stem.replace("single_cond", "")] = n_train_of(p)
        for p in sorted(data_dir.glob("cross_*.npz")):
            cross_n[p.stem.replace("cross_", "").replace("_to_", "->")] = n_train_of(p)

        for (ds, m), inc in s.items():
            if m != model:
                continue
            tgt = cond_of_single(ds)
            if tgt is None or not (inc and inc > 0):
                continue
            cells = []
            for (task, m2), cx in c.items():
                if m2 != model:
                    continue
                pt = parse_task(task)
                if not pt or pt[1] != tgt:
                    continue
                cells.append({"task": task, "R": cx / inc, "dRMSE": cx - inc,
                              "ratio": cross_n.get(task, float("nan")) / single_n.get(tgt, float("nan"))})
            if not cells:
                continue
            units[(fold, tgt)] = {
                "R": med([x["R"] for x in cells]),
                "dRMSE": med([x["dRMSE"] for x in cells]),
                "ratio": med([x["ratio"] for x in cells]),
                "cells": cells,
                "K": {t: single_n.get(t) for t in sorted(single_n)},
            }
    return units


# --------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="se_resnet_bigru")
    ap.add_argument("--metric", default="uncorrected")
    ap.add_argument("--unmatched-dirs", nargs="+", default=UNMATCHED)
    ap.add_argument("--matched-dirs", nargs="+", default=MATCHED)
    ap.add_argument("--unmatched-data", nargs="+", default=None,
                    help="data dirs for the unmatched arms (default: derive from --unmatched-dirs)")
    ap.add_argument("--matched-data", nargs="+", default=None,
                    help="data dirs for the matched arms (default: derive from --matched-dirs)")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-earlystop-audit", action="store_true",
                    help="skip reading best_epoch off the checkpoints")
    ap.add_argument("--md", default="docs/vm_vs_unmatched.md")
    ap.add_argument("--json", default="artifacts/reports/vm_vs_unmatched.json")
    args = ap.parse_args()

    data_un = args.unmatched_data or [derive_data_dir(d, "matched_fold", "data_fold")
                                      for d in args.unmatched_dirs]
    data_vm = args.matched_data or [derive_data_dir(d, "vm_fold", "data_vm_fold")
                                    for d in args.matched_dirs]
    U = build_units(args.unmatched_dirs, data_un, args.model, args.metric)
    V = build_units(args.matched_dirs, data_vm, args.model, args.metric)
    keys = sorted(set(U) & set(V))
    if not keys:
        print("!! no overlapping units -- have the vm runs finished?")
        print("   python scripts/29_run_volume_matched.py")
        return 2

    rows = []
    for k in keys:
        rows.append({
            "fold": k[0], "target": k[1],
            "R_un": U[k]["R"], "R_vm": V[k]["R"], "dR": V[k]["R"] - U[k]["R"],
            "d_un": U[k]["dRMSE"], "d_vm": V[k]["dRMSE"],
            "ratio_un": U[k]["ratio"], "ratio_vm": V[k]["ratio"],
        })

    L = []
    J = L.append
    J("# Volume-matched vs unmatched `R_cross` (Stage 3.2)")
    J("")
    J(f"model `{args.model}`, caliber `{args.metric}` (the only cross-protocol-comparable one, A9).")
    J("Unit = `(fold, target)` = `(condition, bearing)`, n = 15.")
    J("`R_cross = RMSE_cross / RMSE_single` on the byte-identical test bearing; `dRMSE` in minutes.")
    J("`ratio` = median(cross/single `n_train`) before capping; it must be 1.00 after.")
    J("")
    J("| unit | R unmatched | R volume-matched | dR (matched-unmatched) | dRMSE un (min) | dRMSE vm (min) | ratio un | ratio vm |")
    J("|---|---|---|---|---|---|---|---|")
    for r in rows:
        J(f"| ({r['fold']}, cond{r['target']}) | {fmt(r['R_un'])} | {fmt(r['R_vm'])} | "
          f"{r['dR']:+.3f} | {r['d_un']:+.2f} | {r['d_vm']:+.2f} | "
          f"{fmt(r['ratio_un'], 2)}x | {fmt(r['ratio_vm'], 2)}x |")
    J("")

    def blk(name, vals):
        ci = boot_ci(vals, args.n_boot, seed=args.seed)
        return (f"**{name}**: median = **{fmt(med(vals))}** "
                f"(95% CI [{fmt(ci[0])}, {fmt(ci[1])}]), IQR {fmt(iqr(vals))}, "
                f"range {fmt(min(vals))}-{fmt(max(vals))}.")

    Ru = [r["R_un"] for r in rows]
    Rv = [r["R_vm"] for r in rows]
    dR = [r["dR"] for r in rows]
    J("## Overall (n = 15)")
    J("")
    J("- " + blk("R unmatched", Ru))
    J("- " + blk("R volume-matched", Rv))
    J("- " + blk("paired dR (matched - unmatched)", dR))
    J("")
    J(f"- units where the matched design is *lower* (cross does relatively better): "
      f"**{sum(1 for x in dR if x < 0)}/{len(dR)}**")
    J(f"- cross better than single (R < 1): unmatched **{sum(1 for x in Ru if x < 1)}/{len(Ru)}**, "
      f"matched **{sum(1 for x in Rv if x < 1)}/{len(Rv)}**")
    p = wilcoxon(dR)
    J(f"- paired Wilcoxon on dR: {'p = %.3g' % p if p is not None else 'n/a'} "
      f"(n = 15, **indicative only** -- distribution + CI is the evidence)")
    J("")

    J("## Stratified by target condition (the decisive read-out)")
    J("")
    J("| target | n | R unmatched | R matched | dR | ratio un (median) | ratio vm |")
    J("|---|---|---|---|---|---|---|")
    strat = {}
    for t in sorted({r["target"] for r in rows}):
        sub = [r for r in rows if r["target"] == t]
        mu = med([r["R_un"] for r in sub])
        mv = med([r["R_vm"] for r in sub])
        ru = med([r["ratio_un"] for r in sub])
        rv = med([r["ratio_vm"] for r in sub])
        strat[t] = {"n": len(sub), "R_unmatched": mu, "R_matched": mv,
                    "dR": mv - mu, "ratio_unmatched": ru, "ratio_matched": rv,
                    "ci_matched": boot_ci([r["R_vm"] for r in sub], args.n_boot, seed=args.seed + 3)}
        J(f"| cond{t} | {len(sub)} | {fmt(mu)} | {fmt(mv)} | {mv - mu:+.3f} | "
          f"{fmt(ru, 2)}x | {fmt(rv, 2)}x |")
    J("")

    # ------------------------------------------------- residual asymmetry audit
    es = {}
    if not args.no_earlystop_audit:
        J("## Residual asymmetry: early-stopping behaviour (val split is NOT matched)")
        J("")
        J("The design equalises the TRAIN split only; the val split keeps its original size,")
        J("and those sizes differ a lot between arms. Reading `best_epoch` off the")
        J("checkpoints shows how far the two arms' stopping points actually diverge.")
        J("")
        J("| design | arm | n ckpt | median best_epoch | median epochs_run |")
        J("|---|---|---|---|---|")
        for tag, dirs in (("unmatched", args.unmatched_dirs), ("volume-matched", args.matched_dirs)):
            agg = {"single": [], "cross": []}
            for d in dirs:
                st = earlystop_stats(d, args.model)
                for arm, v in st.items():
                    agg[arm].append(v)
            es[tag] = {}
            for arm in ("single", "cross"):
                if not agg[arm]:
                    continue
                n = sum(v["n"] for v in agg[arm])
                mb = med([v["median_best_epoch"] for v in agg[arm]])
                me = med([v["median_epochs_run"] for v in agg[arm]])
                es[tag][arm] = {"n_ckpt": n, "median_best_epoch": mb,
                                "median_epochs_run": me}
                J(f"| {tag} | {arm} | {n} | {fmt(mb, 1)} | {fmt(me, 1)} |")
        J("")

    out = {"model": args.model, "metric": args.metric, "units": rows,
           "earlystop_audit": es,
           "overall": {"median_R_unmatched": med(Ru), "median_R_matched": med(Rv),
                       "median_dR": med(dR), "ci_dR": boot_ci(dR, args.n_boot, seed=args.seed),
                       "n_lower_matched": sum(1 for x in dR if x < 0),
                       "n_cross_better_unmatched": sum(1 for x in Ru if x < 1),
                       "n_cross_better_matched": sum(1 for x in Rv if x < 1),
                       "p_dR_indicative": p},
           "by_target": strat}

    md = ROOT / args.md
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    js = ROOT / args.json
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"wrote {md}")
    print(f"wrote {js}")
    print("\n".join(L[:40]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
