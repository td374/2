"""Reproducibility probe -- measure the pipeline's run-to-run spread.

Why this is a deliverable and not a debugging aid
-------------------------------------------------
Every table in this project is a mean over five seeds, and the paper reasons
about differences of a few minutes. That reasoning is only sound if the
pipeline reproduces itself. It does not: re-running one identical
configuration changes a single-task RMSE by up to a quarter, because the
seeding path requests deterministic algorithms with ``warn_only=True`` and the
models fall back to atomic (order-dependent) GPU reductions -- see
``rulbench/utils.seed_everything``.

The probe makes that spread measurable instead of assumed. It re-runs one
(config, model, data-subset) ``--repeats`` times into fresh output directories
and reports the spread, so any claim thinner than the spread can be withdrawn
before a reviewer withdraws it for us.

    python scripts/17_reproducibility_probe.py \
        --config configs/cross_linear_uda_off.yaml \
        --data-dir artifacts/_smoke_data --models cnn \
        --repeats 4 --out-root artifacts/repro
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import statistics
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]


def _rmse_rows(out_dir: pathlib.Path) -> dict[str, float]:
    path = out_dir / "results_corrected.csv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return {r["dataset"]: float(r["rmse"]) for r in csv.DictReader(fh)
                if r.get("rmse")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--runner", default="scripts/03_run_cross.py")
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--out-root", default="artifacts/repro")
    args = ap.parse_args()

    out_root = REPO / args.out_root
    per_repeat: list[dict[str, float]] = []

    for i in range(args.repeats):
        out = out_root / f"repeat{i + 1}"
        cmd = [
            sys.executable, str(REPO / args.runner), "--config", args.config,
            "--set", "experiments.cross.models=" + ",".join(args.models),
            "--out", str(out.relative_to(REPO)),
        ]
        if args.data_dir:
            cmd += ["--set", "paths.data_dir=" + args.data_dir]
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
        print(f"[probe] repeat{i + 1}/{args.repeats} exit={proc.returncode} "
              f"wall={time.time() - t0:.1f}s")
        if proc.returncode != 0:
            print("\n".join((proc.stderr or "").strip().splitlines()[-12:]))
            return proc.returncode
        per_repeat.append(_rmse_rows(out))

    if not per_repeat or not per_repeat[0]:
        print("[probe] no results collected")
        return 2

    tasks = sorted(per_repeat[0])
    print()
    print(f"run-to-run spread over {args.repeats} identical repeats "
          f"(config={args.config}, models={','.join(args.models)})")
    print(f"{'task':<12}{'mean':>10}{'sd':>10}{'min':>10}{'max':>10}{'spread%':>10}")
    spread_pcts = []
    for t in tasks:
        vals = [r[t] for r in per_repeat if t in r]
        if len(vals) < 2:
            continue
        mean = statistics.fmean(vals)
        sd = statistics.stdev(vals)
        rel = 100.0 * (max(vals) - min(vals)) / mean if mean else float("nan")
        spread_pcts.append(rel)
        print(f"{t:<12}{mean:>10.2f}{sd:>10.2f}{min(vals):>10.2f}{max(vals):>10.2f}{rel:>9.1f}%")

    grand = [statistics.fmean([r[t] for r in per_repeat if t in r]) for t in tasks]
    if len(grand) >= 2:
        gm = statistics.fmean(grand)
        print(f"\nacross-task mean RMSE: {gm:.2f}  "
              f"sd {statistics.stdev(grand):.2f}  "
              f"spread {100.0 * (max(grand) - min(grand)) / gm:.1f}%")
    if spread_pcts:
        print(f"median per-task spread: {statistics.median(spread_pcts):.1f}%")
    print("\nAny effect smaller than this spread must not be claimed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
