"""Stage-5 protocol probe: is a checkpoint set safe to re-evaluate?

    python scripts/34_stage5_protocol_probe.py

Two checks:
  (A) A10 / B8: re-evaluating an already-fixed checkpoint set (s2_chain) must
      reproduce its stored RMSE bit-for-bit on the CURRENT eval code.
  (B) The opposite: re-evaluating a PRE-fix checkpoint set (single_linear) must
      NOT match its stored RMSE, proving that "old checkpoint + new eval" is an
      invalid mix and that the old CSVs cannot be reused.

Gates Stage 5 (see audit/STAGE5_PROTOCOL_CHECK.md §4.1).
"""
import csv
import sys
import pathlib

REPO = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from rulbench.config import Config
from rulbench.data.build import load_dataset
from rulbench.evaluation.report import evaluate_dataset
from rulbench.utils import resolve_device

ART = REPO / "artifacts"


def stored_rmse(root, model="se_resnet_bigru"):
    p = root / "results_corrected.csv"
    with p.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["dataset"] == "cond1":
                return float(row["rmse"])
    return float("nan")


def main():
    dev = resolve_device("cuda")
    cfg = Config.load(str(REPO / "configs" / "base.yaml"),
                      str(REPO / "configs" / "linear.yaml")).freeze()
    seeds = [42, 123, 456, 789, 1011]
    art = load_dataset(ART / "data_linear" / "single_cond1.npz")

    cases = [
        ("A s2_chain/linear  (post-fix, expected MATCH)", ART / "reports" / "s2_chain" / "linear" / "checkpoints", ART / "reports" / "s2_chain" / "linear"),
        ("B single_linear    (pre-fix,  expected MISMATCH)", ART / "reports" / "single_linear" / "se_resnet_bigru" / "checkpoints", ART / "reports" / "single_linear" / "se_resnet_bigru"),
    ]
    for label, run_dir, csv_dir in cases:
        res = evaluate_dataset(art, "se_resnet_bigru", cfg, run_dir, seeds, device=dev)
        got = res["metrics"]["rmse"]
        want = stored_rmse(csv_dir)
        rel = abs(got - want) / (abs(want) + 1e-12) * 100.0
        print(f"{label}\n    re-eval RMSE = {got:.4f}   stored = {want:.4f}   rel diff = {rel:.3f}%")
        # also report picp to show B4 surface
        print(f"    picp = {res['metrics']['picp']:.4f}  mpiw = {res['metrics']['mpiw']:.3f}  crps = {res['metrics']['crps']:.3f}")


if __name__ == "__main__":
    main()
