"""Training-volume audit for the matched cross/single arms.

Why this matters for IP5
------------------------
``configs/cross_matched_linear.yaml`` matches the two arms on the TEST
population, so ``R_cross = RMSE_cross / RMSE_single`` no longer carries the
27x test-composition artifact. It does NOT match the TRAINING-set size: the
source condition's bearings 1-3 can hold far more or far fewer windows than the
target condition's bearings 1-3, because the three XJTU-SY conditions have very
different lifetimes.

So a raw ``R_cross`` conflates two things:

    different condition      (the effect IP5 wants to measure)
  + different training volume (a confound)

and the confound's sign is target-dependent, so it does not cancel in the
aggregate. This script makes the confound visible: for every cross dataset it
prints ``n_train`` and the ratio to the matching in-condition arm.

What to do with it
------------------
* Disclose it (rule A8, extended from the test to the training population).
* If a real volume-matched arm is added later, its ratios should all be 1.00.
  Until then, per-cell ``R_cross`` values are descriptive, not causal.

Usage
-----
    python scripts/22_train_volume_audit.py                        # artifacts/data_matched
    python scripts/22_train_volume_audit.py --data-dir artifacts/data_fold1
    python scripts/22_train_volume_audit.py --json artifacts/reports/volume_audit.json
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def stats(path: pathlib.Path):
    z = np.load(path, allow_pickle=True)
    return {
        "n_train": int(len(z["train/ratio"])),
        "n_val": int(len(z["val/ratio"])),
        "n_test": int(len(z["test/ratio"])),
        "train_bearings": _counts(z["train/bearing_id"]),
        "test_bearings": _counts(z["test/bearing_id"]),
        "test_conditions": _counts(z["test/condition"]),
        "test_lifetime": sorted({float(x) for x in np.unique(z["test/lifetime"])}),
    }


def _counts(arr) -> dict:
    return {int(k): int(v) for k, v in sorted(collections.Counter(arr.tolist()).items())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="artifacts/data_matched")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    d = ROOT / args.data_dir
    if not d.is_dir():
        print(f"!! {d} does not exist")
        return 2

    single = {}
    for c in (1, 2, 3):
        p = d / f"single_cond{c}.npz"
        if p.exists():
            single[str(c)] = stats(p)

    rows = []
    print(f"data dir: {args.data_dir}\n")
    print("in-condition arms")
    print(f"  {'dataset':<16}{'n_train':>9}{'n_val':>7}{'n_test':>8}{'train bearings':>28}")
    for c, s in sorted(single.items()):
        print(f"  {'single_cond'+c:<16}{s['n_train']:>9}{s['n_val']:>7}{s['n_test']:>8}"
              f"{str(s['train_bearings']):>28}")

    print("\ntransfer arms (ratio = n_train / matching in-condition n_train)")
    print(f"  {'dataset':<20}{'n_train':>9}{'n_val':>7}{'n_test':>8}{'target':>8}{'ratio':>9}")
    for p in sorted(glob.glob(str(d / "cross_*.npz"))):
        name = os.path.basename(p)
        s = stats(pathlib.Path(p))
        # 'cross_1+2_to_3.npz' -> target '3'
        tgt = name.split("_to_")[1].replace(".npz", "")
        base = single.get(tgt, {}).get("n_train")
        ratio = (s["n_train"] / base) if base else float("nan")
        rows.append({"dataset": name.replace(".npz", ""), "target": tgt, **s,
                     "ratio_vs_single": ratio})
        print(f"  {name.replace('.npz',''):<20}{s['n_train']:>9}{s['n_val']:>7}"
              f"{s['n_test']:>8}{tgt:>8}{ratio:>8.2f}x")

    ratios = [r["ratio_vs_single"] for r in rows if r["ratio_vs_single"] == r["ratio_vs_single"]]
    if ratios:
        print(f"\ntrain-volume ratio spans {min(ratios):.2f}x - {max(ratios):.2f}x "
              f"({max(ratios) / min(ratios):.0f}x spread)")
        by_tgt = {}
        for r in rows:
            by_tgt.setdefault(r["target"], []).append(r["ratio_vs_single"])
        for t, rs in sorted(by_tgt.items()):
            side = "MORE" if min(rs) > 1 else ("LESS" if max(rs) < 1 else "mixed")
            print(f"  target cond{t}: {min(rs):.2f}x - {max(rs):.2f}x  -> cross arm always {side} data")
        print("\n=> R_cross therefore conflates 'different condition' with 'different training "
              "volume'.\n   Per-cell values are descriptive, not causal; disclose (A8).")

    if args.json:
        out = ROOT / args.json
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"data_dir": args.data_dir, "single": single, "cross": rows},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
