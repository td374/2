"""S3 -- cross-condition transfer: baseline anchors + the A5 on/off verdict.

Two jobs, both of them disclosure work rather than modelling work.

1. Anchors (rules A1/A2). For every cross task this computes the trivial
   baselines a model actually has to clear, evaluated in the *same*
   post-processed space the models are scored in:

   * ``const_one_uncorr``  -- answer ratio 1.0, no correction.
   * ``const_corrected``   -- answer any constant, then apply the median-ratio
     correction. Note the correction divides the constant back out, so **every**
     constant collapses to the same prediction, ``median(val_ratio) * lifetime``.
     That is exactly why ``constant_one`` and ``constant_mean`` come out
     numerically identical in the cross tables -- not a bug, a property.
   * ``oracle_const_task``  -- the best single constant picked with hindsight on
     the target task, closed form (lifetime^2-weighted least squares).
   * ``oracle_const_bearing`` -- the same fitted per bearing: the absolute floor
     for any constant-form predictor on this task.

   If a feature-based model cannot beat ``oracle_const_task`` it has extracted
   nothing a constant could not have supplied.

2. A5 verdict. Compare the transductive arm (MK-MMD on) against the source-only
   control (``uda.enabled: false``). The paired per-task gap is what separates
   "the adaptation helped" from "the model saw the target data".

    python scripts/15_cross_transfer.py \
        --uda   artifacts/reports/cross_uda_linear \
        --src   artifacts/reports/cross_source_only_linear \
        --data-dir artifacts/data_linear \
        --out   artifacts/reports/cross_transfer_summary.json \
        --table paper/tables/cross_transfer.md
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from typing import Dict, List, Optional

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rulbench.data.build import load_dataset  # noqa: E402
from rulbench.evaluation.metrics import evaluate_predictions  # noqa: E402

_EPS = 1e-12


# ----------------------------------------------------------------- anchors
def _rmse(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(true, float) - np.asarray(pred, float)) ** 2)))


def _oracle_constant(ratio: np.ndarray, lifetime: np.ndarray) -> float:
    """Best constant ratio under the scored (minute-space) error.

    Minimising ``sum_i L_i^2 (c - r_i)^2`` over ``c`` gives the L^2-weighted mean
    of the ratios; the ``clip(., 0, L)`` in the scoring path is inactive for
    ``c <= 1`` and forces ``c = 1`` above that.
    """
    w = np.asarray(lifetime, float) ** 2
    c = float(np.sum(w * np.asarray(ratio, float)) / (np.sum(w) + _EPS))
    return min(max(c, 0.0), 1.0)


def anchors_for(art: Dict) -> Dict[str, float]:
    test = art["splits"]["test"]
    ratio = np.asarray(test["ratio"], float)
    life = np.asarray(test["lifetime"], float)
    truth = ratio * life
    bids = np.asarray(test["bearing_id"])

    val_ratio_med = float(np.median(np.asarray(art["splits"]["val"]["ratio"], float)))

    # (a) answer 1.0, no correction: prediction == lifetime
    pred_one = np.clip(1.0 * life, 0.0, life)

    # (b) any constant + median-ratio correction == median(val_ratio) * lifetime
    pred_corr = np.clip(val_ratio_med * life, 0.0, life)

    # (c) oracle constant, pooled over the task
    c_task = _oracle_constant(ratio, life)
    pred_task = np.clip(c_task * life, 0.0, life)

    # (d) oracle constant, fitted separately per bearing (hardest floor)
    pred_bearing = np.empty_like(truth)
    for bid in np.unique(bids):
        m = bids == bid
        c_b = _oracle_constant(ratio[m], life[m])
        pred_bearing[m] = np.clip(c_b * life[m], 0.0, life[m])

    out: Dict[str, float] = {
        "n_windows": int(truth.size),
        "n_bearings": int(len(np.unique(bids))),
        "mean_lifetime_min": float(np.mean(life)),
        "median_val_ratio": val_ratio_med,
        "oracle_const_ratio_task": c_task,
    }
    for tag, pred in (
        ("const_one_uncorr", pred_one),
        ("const_corrected", pred_corr),
        ("oracle_const_task", pred_task),
        ("oracle_const_bearing", pred_bearing),
    ):
        met = evaluate_predictions(truth, pred)
        out[f"{tag}__rmse"] = met["rmse"]
        out[f"{tag}__nrmse"] = met["nrmse"]
        out[f"{tag}__phm_official"] = met["phm_official"]
    return out


# ------------------------------------------------------------------- tables
def read_csv(path: pathlib.Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _by_dataset(rows: List[Dict[str, str]], metric: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for r in rows:
        try:
            out.setdefault(r["dataset"], {})[r["model"]] = float(r[metric])
        except (KeyError, ValueError):
            continue
    return out


def _fmt(x: Optional[float], nd: int = 2) -> str:
    return "-" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def _wilcoxon_p(values: List[float]) -> float:
    """Paired Wilcoxon p-value; NaN when scipy is absent or n is too small."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return float("nan")
    arr = np.asarray(values, float)
    if arr.size < 6 or np.allclose(arr, 0.0):
        return float("nan")
    try:
        return float(wilcoxon(arr).pvalue)
    except ValueError:
        return float("nan")


def build_table(
    anchors: Dict[str, Dict[str, float]],
    uda: Dict[str, Dict[str, float]],
    src: Dict[str, Dict[str, float]],
    models: List[str],
) -> str:
    tasks = sorted(anchors)
    lines: List[str] = []
    lines.append("| task | oracle const (task) | oracle const (bearing) | const-1 uncorr "
                 "| corrected const | " + " | ".join(f"{m} (UDA)" for m in models) + " | "
                 + " | ".join(f"{m} (src-only)" for m in models) + " |")
    lines.append("|" + "---|" * (5 + 2 * len(models)))
    for t in tasks:
        a = anchors[t]
        cells = [
            t,
            _fmt(a.get("oracle_const_task__rmse")),
            _fmt(a.get("oracle_const_bearing__rmse")),
            _fmt(a.get("const_one_uncorr__rmse")),
            _fmt(a.get("const_corrected__rmse")),
        ]
        cells += [_fmt(uda.get(t, {}).get(m)) for m in models]
        cells += [_fmt(src.get(t, {}).get(m)) for m in models]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="artifacts/data_linear")
    ap.add_argument("--uda", default="artifacts/reports/cross_uda_linear")
    ap.add_argument("--src", default="artifacts/reports/cross_source_only_linear")
    ap.add_argument("--out", default="artifacts/reports/cross_transfer_summary.json")
    ap.add_argument("--table", default="paper/tables/cross_transfer.md")
    ap.add_argument("--models", nargs="+",
                    default=["se_resnet_bigru", "cnn", "lstm", "gru"])
    # Only these carry an ``mmd_representation`` head, so only for these does the
    # transductive arm differ from the source-only control at all. For a CNN /
    # LSTM / GRU the two arms are identical by construction, and quoting a
    # "gain" for them would be quoting noise.
    ap.add_argument("--a5-models", nargs="+",
                    default=["se_resnet_bigru", "pure_bigru"])
    args = ap.parse_args()

    data_dir = REPO / args.data_dir
    npz = sorted(p for p in data_dir.glob("cross_*.npz"))
    if not npz:
        print(f"[error] no cross_*.npz under {data_dir}")
        return 2

    anchors: Dict[str, Dict[str, float]] = {}
    for p in npz:
        art = load_dataset(p)
        name = art["name"]
        anchors[name] = anchors_for(art)
        a = anchors[name]
        print(f"[anchor] {name:<10} "
              f"oracle_task {a['oracle_const_task__rmse']:8.2f}  "
              f"oracle_bearing {a['oracle_const_bearing__rmse']:8.2f}  "
              f"const1_uncorr {a['const_one_uncorr__rmse']:8.2f}  "
              f"const_corrected {a['const_corrected__rmse']:8.2f}")

    uda_rows = read_csv(REPO / args.uda / "results_corrected.csv")
    src_rows = read_csv(REPO / args.src / "results_corrected.csv")
    uda_u = _by_dataset(read_csv(REPO / args.uda / "results_uncorrected.csv"), "rmse")
    src_u = _by_dataset(read_csv(REPO / args.src / "results_uncorrected.csv"), "rmse")
    uda = _by_dataset(uda_rows, "rmse")
    src = _by_dataset(src_rows, "rmse")

    table = build_table(anchors, uda, src, args.models)
    tbl_path = REPO / args.table
    tbl_path.parent.mkdir(parents=True, exist_ok=True)
    tbl_path.write_text(
        "# S3 cross-condition transfer (linear label, XJTU-SY)\n\n"
        "All RMSE in minutes, corrected (median-ratio) unless the column says "
        "otherwise. `oracle const` = best constant chosen with hindsight on the "
        "target, i.e. the floor a feature-free predictor can reach.\n\n"
        + table + "\n",
        encoding="utf-8",
    )

    # ------------------------------------------------------------- A5 verdict
    # Two variants. ``all9`` is the plain paired comparison. ``warm`` drops the
    # tasks trained first in each process, because scripts/17_reproducibility_probe
    # showed the pipeline reproduces bit-exactly for most (task, model) pairs but
    # can drift on the earliest-trained ones. If the sign survives both, the
    # verdict does not hinge on that drift.
    warmup = sorted(anchors)[:2]
    verdict: Dict[str, object] = {}
    for m in args.a5_models:
        row: Dict[str, object] = {}
        for name, skip in (("all", ()), ("excl_warmup", tuple(warmup))):
            deltas = []
            for t in sorted(anchors):
                if t in skip:
                    continue
                a, b = uda.get(t, {}).get(m), src.get(t, {}).get(m)
                if a is not None and b is not None:
                    deltas.append(b - a)   # >0 means UDA-on is better
            if not deltas:
                continue
            d = np.asarray(deltas, float)
            row[name] = {
                "n_tasks": int(d.size),
                "mean_gain_uda_on_min": float(d.mean()),
                "median_gain_uda_on_min": float(np.median(d)),
                "tasks_better_with_uda": int((d > 0).sum()),
                "tasks_better_source_only": int((d < 0).sum()),
                "paired_wilcoxon_p": float(_wilcoxon_p(d.tolist())),
            }
        if row:
            row["excluded_tasks"] = list(warmup)
            verdict[m] = row

    # sanity check: for a model without an alignment head the two arms must agree.
    # They only do so within one process (see scripts/17_reproducibility_probe), so
    # a mismatch here is a measure of cross-process drift, not of MK-MMD.
    inert_check: Dict[str, Dict[str, object]] = {}
    for m in args.models:
        if m in args.a5_models:
            continue
        pairs = [(t, uda.get(t, {}).get(m), src.get(t, {}).get(m)) for t in sorted(anchors)]
        pairs = [(t, a, b) for t, a, b in pairs if a is not None and b is not None]
        if not pairs:
            continue
        worst_t, worst = max(((t, abs(a - b)) for t, a, b in pairs), key=lambda kv: kv[1])
        rel = 100.0 * worst / max(1e-9, np.mean([a for _, a, _ in pairs]))
        inert_check[m] = {
            "n_tasks": len(pairs),
            "max_abs_diff_min": float(worst),
            "worst_task": worst_t,
            "max_rel_diff_pct": float(rel),
            "identical": bool(worst < 1e-6),
        }

    # beat-the-oracle census on the transductive arm
    census: Dict[str, Dict[str, int]] = {}
    for m in args.models:
        w_o = w_c = n = 0
        for t in sorted(anchors):
            v = uda.get(t, {}).get(m)
            if v is None:
                continue
            n += 1
            w_o += int(v < anchors[t]["oracle_const_task__rmse"])
            w_c += int(v < anchors[t]["const_corrected__rmse"])
        census[m] = {"n_tasks": n, "beats_oracle_const": w_o, "beats_corrected_const": w_c}

    summary = {
        "anchors": anchors,
        "uda_on": uda,
        "source_only": src,
        "uda_on_uncorrected": uda_u,
        "source_only_uncorrected": src_u,
        "a5_verdict": verdict,
        "a5_inert_arm_check": inert_check,
        "beat_oracle_census": census,
    }
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(table)
    print()
    print("[A5] mean gain of MK-MMD-on over source-only (minutes, >0 favours UDA):")
    for m, row in verdict.items():
        for variant in ("all", "excl_warmup"):
            v = row.get(variant)
            if not v:
                continue
            tag = "all 9 tasks " if variant == "all" else "excl. warm-up"
            print(f"  {m:<18} [{tag}] {v['mean_gain_uda_on_min']:+8.2f}  "
                  f"({v['tasks_better_with_uda']}W/{v['tasks_better_source_only']}L of {v['n_tasks']})  "
                  f"wilcoxon p={v['paired_wilcoxon_p']:.3f}")
    if inert_check:
        print("  [check] models with no alignment head (arms should agree within a process):")
        for m, v in inert_check.items():
            print(f"    {m:<16} max|diff| {v['max_abs_diff_min']:.2f} min "
                  f"({v['max_rel_diff_pct']:.1f}%) on {v['worst_task']}  "
                  f"identical={v['identical']}")
    print()
    print("[A2] does the proposed model clear the hindsight constant floor?")
    for m, v in census.items():
        print(f"  {m:<18} beats oracle const on {v['beats_oracle_const']}/{v['n_tasks']} tasks, "
              f"beats corrected const on {v['beats_corrected_const']}/{v['n_tasks']}")
    print(f"\nwrote {out_path.relative_to(REPO)} and {tbl_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
