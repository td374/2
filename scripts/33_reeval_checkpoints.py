"""Re-evaluate an existing report directory with the CURRENT code, from checkpoints.

Why (Stage 3 audit finding B9)
------------------------------
``artifacts/reports/matched_fold{k}`` -- the Stage 3.1 / 3.3 baseline -- was written
in two sittings:

    matched_fold{k}/single/results_*.csv   10:51-12:09   (before the B8 fix)
    matched_fold{k}/cross/results_*.csv    16:25-16:30   (after  the B8 fix)

``rulbench/evaluation/inference.py`` (B8: seed the MC-Dropout eval path) was
patched at **16:21**. So the baseline's ``R_cross = RMSE_cross / RMSE_single``
divides a post-B8 numerator by a pre-B8 denominator. The offset is small
(~0.1-0.2%, the measured B8 magnitude) but it is a real vintage mismatch, and it
also contaminates any paired comparison against a fully post-B8 run.

The fix is a re-evaluation, not a retrain: per disclosure rule A10 the prediction
is a pure function of ``(checkpoint, data, seed)`` once the eval path is seeded, so
re-scoring the existing checkpoints reproduces the full train->eval numbers
bit-for-bit. This script therefore writes a **new** directory (default
``<src>_b8``) and leaves the original untouched; pass ``--in-place`` to overwrite.

It also prints the delta against the source CSVs, which is the audit evidence for
how large the vintage effect actually was.

Usage
-----
    python scripts/33_reeval_checkpoints.py \
        --src-report artifacts/reports/matched_fold1 \
        --data-dir artifacts/data_fold1 \
        --config configs/matched_folds/fold1.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rulbench.config import Config, as_list  # noqa: E402
from rulbench.data.build import load_dataset  # noqa: E402
from rulbench.evaluation.report import (  # noqa: E402
    evaluate_dataset,
    results_to_records,
    save_table,
    summary_by_model,
)
from rulbench.models.registry import is_trivial, resolve  # noqa: E402
from rulbench.pipelines.run import evaluate_trivial  # noqa: E402
from rulbench.utils import jsonable, resolve_device  # noqa: E402


def load_csv_rmse(path: pathlib.Path) -> dict:
    out = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("dataset") and r.get("model") and r.get("rmse"):
                try:
                    out[(r["dataset"], r["model"])] = float(r["rmse"])
                except ValueError:
                    pass
    return out


def reeval_arm(cfg, data_dir: pathlib.Path, src_arm: pathlib.Path, dst_arm: pathlib.Path,
               models, seeds, device, pattern: str) -> dict:
    paths = sorted(data_dir.glob(pattern))
    if not paths:
        print(f"   !! no {pattern} under {data_dir}")
        return {}
    results = []
    for ds_path in paths:
        artifact = load_dataset(ds_path)
        for model_name in models:
            if is_trivial(model_name):
                results.append(evaluate_trivial(artifact, model_name, cfg))
                continue
            resolve(model_name)
            results.append(evaluate_dataset(artifact, model_name, cfg,
                                            src_arm / "checkpoints", seeds, device=device))

    dst_arm.mkdir(parents=True, exist_ok=True)
    rec_c = results_to_records(results, corrected=True)
    rec_u = results_to_records(results, corrected=False)
    save_table(rec_c, dst_arm / "results_corrected.csv")
    save_table(rec_u, dst_arm / "results_uncorrected.csv")
    save_table(summary_by_model(results, metric=cfg.get_path("report.headline_metric", "rmse")),
               dst_arm / "summary_by_model.csv")
    with (dst_arm / "results_full.json").open("w", encoding="utf-8") as fh:
        json.dump([jsonable({k: v for k, v in r.items() if k != "predictions"})
                   for r in results], fh, indent=2, ensure_ascii=False, default=str)

    # vintage audit: new rmse vs the source csv
    old = load_csv_rmse(src_arm / "results_uncorrected.csv")
    new = {(r["dataset"], r["model"]): r["rmse"] for r in rec_u}
    deltas = []
    for k, v in new.items():
        if k in old and old[k]:
            deltas.append(100.0 * (v - old[k]) / old[k])
    stats = {}
    if deltas:
        stats = {"n": len(deltas), "mean_pct": sum(deltas) / len(deltas),
                 "max_abs_pct": max(abs(d) for d in deltas),
                 "n_identical": sum(1 for d in deltas if abs(d) < 1e-9)}
    return {"recorded": new, "delta_vs_source": stats}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-report", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dst-report", default=None, help="default: <src-report>_b8")
    ap.add_argument("--in-place", action="store_true", help="overwrite the source report")
    ap.add_argument("--models", nargs="+",
                    default=["se_resnet_bigru", "constant_one", "constant_mean", "ridge"])
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--arms", nargs="+", default=["single", "cross"])
    ap.add_argument("--json", default=None, help="where to append the audit record")
    args = ap.parse_args()

    cfg = Config.load("configs/base.yaml", args.config).freeze()
    seeds = args.seeds or [int(s) for s in as_list(
        cfg.get_path("runtime.seeds") or [42, 123, 456, 789, 1011])]
    device = resolve_device(cfg.get_path("runtime.device", "cuda"))

    src = ROOT / args.src_report
    dst = src if args.in_place else ROOT / (args.dst_report or f"{args.src_report}_b8")
    data_dir = ROOT / args.data_dir
    print(f"[reeval] {src.name} -> {dst.name}  seeds={seeds}  device={device}")

    report = {"src": args.src_report, "dst": str(dst.relative_to(ROOT)), "seeds": seeds, "arms": {}}
    for arm in args.arms:
        t0 = time.time()
        pattern = "single_cond*.npz" if arm == "single" else "cross_*.npz"
        info = reeval_arm(cfg, data_dir, src / arm, dst / arm, args.models, seeds, device, pattern)
        report["arms"][arm] = info.get("delta_vs_source", {})
        d = info.get("delta_vs_source", {})
        print(f"[reeval] {arm:<7} wall={time.time() - t0:.1f}s  vintage delta vs source: "
              f"n={d.get('n')} mean={d.get('mean_pct', 0):+.4f}% "
              f"max|.|={d.get('max_abs_pct', 0):.4f}% identical={d.get('n_identical')}")

    print(f"\n[reeval] wrote {dst}")
    if args.json:
        out = ROOT / args.json
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"[reeval] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
