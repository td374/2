"""Drive the Stage 3.2 volume-matched runs (single + cross) for every fold.

Companion to ``scripts/28_make_volume_matched.py`` (which builds the capped
datasets and the configs). This one just executes

    configs/matched_folds_vm/fold{k}.yaml
        -> artifacts/reports/vm_fold{k}/single    (02_run_single.py)
        -> artifacts/reports/vm_fold{k}/cross     (03_run_cross.py)

for k = 1..5. Checkpoints are named ``<dataset>__<model>__seed<seed>.pt`` and
skipped when present, so the run is resumable: re-invoking after an interruption
finishes only what is missing.

    python scripts/29_run_volume_matched.py
    python scripts/29_run_volume_matched.py --folds 1
    python scripts/29_run_volume_matched.py --folds 1 --skip-cross
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]


def run(cmd: list[str], label: str) -> bool:
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    ok = proc.returncode == 0
    print(f"[vm-run] {label:<26} exit={proc.returncode} wall={time.time() - t0:.1f}s", flush=True)
    if not ok:
        tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-15:])
        print(tail, flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--configs-dir", default="configs/matched_folds_vm")
    ap.add_argument("--reports-dir", default="artifacts/reports")
    ap.add_argument("--skip-single", action="store_true")
    ap.add_argument("--skip-cross", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    for k in args.folds:
        cfg = f"{args.configs_dir}/fold{k}.yaml"
        base = f"{args.reports_dir}/vm_fold{k}"
        if not (REPO / cfg).exists():
            print(f"!! {cfg} missing -- run scripts/28_make_volume_matched.py first", flush=True)
            return 2
        if not args.skip_single:
            if not run([py, "scripts/02_run_single.py", "--config", cfg,
                        "--set", f"paths.report_dir={base}/single"], f"single fold{k}"):
                return 1
        if not args.skip_cross:
            if not run([py, "scripts/03_run_cross.py", "--config", cfg,
                        "--set", f"paths.report_dir={base}/cross"], f"cross fold{k}"):
                return 1

    print("\n[vm-run] done. Aggregate with scripts/23_stat_unit_rcross.py "
          "--fold artifacts/reports/vm_fold1 ...", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
