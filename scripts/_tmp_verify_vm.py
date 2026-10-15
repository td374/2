"""Throwaway: verify the volume-matched datasets against the pilot folds."""
from __future__ import annotations

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
design = json.loads((ROOT / "artifacts/reports/volume_matched_design.json").read_text(encoding="utf-8"))
K = {(d["fold"], d["target"]): d["K"] for d in design}

lines = []
bad = 0
for f in range(1, 6):
    src = ROOT / f"artifacts/data_fold{f}"
    dst = ROOT / f"artifacts/data_vm_fold{f}"
    for p in sorted(dst.glob("*.npz")):
        stem = p.stem
        tgt = stem.replace("single_cond", "") if stem.startswith("single_cond") else stem.split("_to_")[1]
        Kf = K[(f, tgt)]
        zs = np.load(src / f"{stem}.npz", allow_pickle=True)
        zd = np.load(p, allow_pickle=True)
        ok_train = len(zd["train/ratio"]) == Kf
        # val/test must be byte-identical
        ok_val = np.array_equal(zs["val/X"], zd["val/X"]) and np.array_equal(zs["val/ratio"], zd["val/ratio"])
        ok_test = np.array_equal(zs["test/X"], zd["test/X"]) and np.array_equal(zs["test/ratio"], zd["test/ratio"])
        # every kept window must exist in the original train set (exact X match)
        orig = set(map(lambda r: r.tobytes(), zs["train/X"]))
        kept = sum(1 for r in zd["train/X"] if r.tobytes() in orig)
        ok_subset = kept == len(zd["train/X"])
        # per-bearing counts
        bc = {int(b): int(c) for b, c in sorted(zip(*np.unique(zd["train/bearing_id"], return_counts=True)))}
        flag = "OK " if (ok_train and ok_val and ok_test and ok_subset) else "BAD"
        if flag == "BAD":
            bad += 1
        lines.append(f"{flag} f{f} {stem:<18} n_train={len(zd['train/ratio']):>4} (K={Kf:>4}) "
                     f"val_same={ok_val} test_same={ok_test} subset={ok_subset} bear={bc}")

lines.append(f"\nTOTAL bad = {bad}")
txt = "\n".join(lines)
(ROOT / "artifacts/reports/_tmp_verify_vm.txt").write_text(txt, encoding="utf-8")
print(txt)
