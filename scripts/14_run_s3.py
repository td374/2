"""Batched driver for the S3 cross-condition experiments.

Why this exists
---------------
``run_cross`` writes ``results_corrected.csv`` from *only* the models named in
the current invocation, so a model-at-a-time outer loop would overwrite the
table on every call. At the same time a single all-models invocation for the
cross tasks does not fit in one command window (~9 tasks x 8 models x 5 seeds).

The way out is that ``train_one`` skips a seed whose checkpoint already exists
and is keyed on ``(dataset, model, seed)``. So an experiment may be chopped into
many short invocations with no bookkeeping at all -- re-running an interrupted
batch simply resumes. This driver only adds (a) a readable invocation, (b) a
wall-clock reading, and (c) a trimmed log tail, so the outer loop can decide
what to run next without parsing a wall of output.

Finish a split experiment with a call that names *every* model: it re-scores
from the existing checkpoints in seconds and writes the complete table.

    python scripts/14_run_s3.py --config configs/cross_linear.yaml \
        --models gru lstm constant_one constant_mean ridge \
        --out artifacts/reports/cross_uda_linear
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="config overlay, e.g. configs/cross_linear.yaml")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", default=None, help="default: every seed in runtime.seeds")
    ap.add_argument("--data-dir", default=None, help="override paths.data_dir (task subsetting)")
    ap.add_argument("--out", required=True, help="output dir shared by every batch of this experiment")
    ap.add_argument("--runner", default="scripts/03_run_cross.py")
    ap.add_argument("--tail", type=int, default=16)
    args = ap.parse_args()

    cmd = [
        sys.executable,
        str(REPO / args.runner),
        "--config", args.config,
        "--set", "experiments.cross.models=" + ",".join(args.models),
        "--out", args.out,
    ]
    if args.seeds:
        cmd += ["--set", "runtime.seeds=" + ",".join(args.seeds)]
    if args.data_dir:
        cmd += ["--set", "paths.data_dir=" + args.data_dir]

    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    elapsed = time.time() - started

    seeds = ",".join(args.seeds) if args.seeds else "all"
    print(f"[s3] models={','.join(args.models)}  seeds={seeds}  "
          f"exit={proc.returncode}  wall={elapsed:.1f}s")
    out_lines = (proc.stdout or "").strip().splitlines()
    for line in out_lines[-args.tail:]:
        print("   ", line)
    if proc.returncode != 0:
        print("--- stderr tail ---")
        for line in (proc.stderr or "").strip().splitlines()[-20:]:
            print("   ", line)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
