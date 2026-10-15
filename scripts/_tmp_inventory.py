"""Throwaway: inventory train/val/test window counts for the matched folds."""
from __future__ import annotations

import glob
import json
import os
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]

rows = []
for k in range(1, 6):
    d = ROOT / f"artifacts/data_fold{k}"
    if not d.is_dir():
        continue
    for p in sorted(d.glob("*.npz")):
        z = np.load(p, allow_pickle=True)
        name = p.stem
        rows.append({
            "fold": k,
            "ds": name,
            "n_train": int(len(z["train/ratio"])),
            "n_val": int(len(z["val/ratio"])),
            "n_test": int(len(z["test/ratio"])),
        })

# matched (pilot, single fold) dir for reference
d = ROOT / "artifacts/data_matched"
if d.is_dir():
    for p in sorted(d.glob("*.npz")):
        z = np.load(p, allow_pickle=True)
        rows.append({"fold": 0, "ds": p.stem,
                     "n_train": int(len(z["train/ratio"])),
                     "n_val": int(len(z["val/ratio"])),
                     "n_test": int(len(z["test/ratio"]))})

out = []
out.append(f"{'fold':>4} {'dataset':<20}{'n_train':>9}{'n_val':>8}{'n_test':>8}")
for r in rows:
    out.append(f"{r['fold']:>4} {r['ds']:<20}{r['n_train']:>9}{r['n_val']:>8}{r['n_test']:>8}")

# per (fold,target) min across arms
out.append("\n--- per (fold,target): K = min(n_train over single + its cross tasks) ---")
out.append(f"{'fold':>4} {'target':>6}{'K':>8}  arms(n_train)")
by = {}
for r in rows:
    if r["fold"] == 0:
        continue
    ds = r["ds"]
    if ds.startswith("single_cond"):
        tgt = ds.replace("single_cond", "")
        by.setdefault((r["fold"], tgt), {})["SINGLE"] = r["n_train"]
    elif ds.startswith("cross_") and "_to_" in ds:
        tgt = ds.split("_to_")[1]
        by.setdefault((r["fold"], tgt), {})[ds] = r["n_train"]

for (f, t), arms in sorted(by.items()):
    K = min(arms.values())
    details = " ".join(f"{k}={v}" for k, v in sorted(arms.items()))
    out.append(f"{f:>4} {t:>6}{K:>8}  {details}")

txt = "\n".join(out)
(ROOT / "artifacts/reports/_tmp_inventory.txt").write_text(txt, encoding="utf-8")
print(txt)
