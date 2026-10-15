"""Windowed trivial-baseline anchors across label definitions.

Supports the two contributions in ``docs/contributions.md`` that hinge on the
label rule:

* **IP3 (baseline ladder / analytic anchors).** For a given label definition the
  *constant* predictor family has closed-form windowed RMSE values. Reporting
  them turns "did the model beat the baseline?" from an empirical contest into a
  comparison against a stated lower bound -- disclosure rules A1/A2.
* **IP4 (label-sensitivity spectrum).** Sweeping the label definition and
  watching how the gap between the best constant and the constant-1.0 predictor
  moves is the causal evidence that *the label rule, not the architecture,
  governs the benchmark's discriminative power*.

Two quantities matter per (bearing, label) pair:

``const_one``   RMSE of the "always predict 1.0" predictor -- the naive bound
                that a piecewise label makes deceptively strong.
``best_const``  RMSE of the *optimal* constant predictor (the mean of the scored
                labels). No constant can do better, so this is the real hurdle.
``LI``          label informativeness ``1 - (best_const / const_one)**2``: the
                share of the constant-1.0 error a constant cannot explain.
                Near 0 means the benchmark is nearly non-discriminative.

Cross-check: the piecewise row whose ``h`` equals the config's own
``data.label.healthy_ratio`` must reproduce the ``test/ratio`` stored in the
prepared artefacts, so a silent change to label construction is caught. Rows
with other ``h`` values are *supposed* to differ and are not flagged.

    python scripts/09_label_baseline_anchors.py
    python scripts/09_label_baseline_anchors.py --config configs/phm2012.yaml --out anchors.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rulbench.config import Config  # noqa: E402
from rulbench.data.labels import linear_rul, piecewise_linear_rul  # noqa: E402

DEFAULT_H = (0.5, 0.6, 0.7, 0.8, 0.9)


def _scored_ends(n_snap: int, window_len: int) -> np.ndarray:
    """Indices of the window's last snapshot; the first window_len-1 never score."""
    return np.arange(window_len - 1, n_snap)


def _anchors(ratio: np.ndarray, life: float) -> dict:
    rmse_one = life * float(np.sqrt(np.mean((ratio - 1.0) ** 2)))
    c_best = float(ratio.mean())          # optimal constant for RMSE
    rmse_best = life * float(np.sqrt(np.mean((ratio - c_best) ** 2)))
    rmse_zero = life * float(np.sqrt(np.mean(ratio**2)))
    li = 1.0 - (rmse_best / rmse_one) ** 2 if rmse_one else float("nan")
    return {
        "const_one": rmse_one,
        "best_const": rmse_best,
        "const_zero": rmse_zero,
        "mean_ratio": c_best,
        "label_informativeness": li,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "base.yaml"))
    ap.add_argument("--h", type=float, nargs="*", default=list(DEFAULT_H),
                    help="healthy_ratio values for the piecewise family")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    ap.add_argument("--data-dir", default=None,
                    help="override paths.data_dir (e.g. a *_linear build)")
    ap.add_argument("--strategy", default=None,
                    help="override data.label.strategy (which row to cross-check)")
    arts = ap.parse_args()

    # Mirror the CLI: the given config is an overlay merged on top of base.yaml.
    base = REPO / "configs" / "base.yaml"
    cfg_path = pathlib.Path(arts.config)
    load_paths = [base] if base.exists() and cfg_path.resolve() != base.resolve() else []
    load_paths.append(cfg_path)
    cfg = Config.load(*load_paths)
    if arts.data_dir:
        cfg = cfg.override(paths__data_dir=arts.data_dir)
    if arts.strategy:
        cfg = cfg.override(data__label__strategy=arts.strategy)
    L = int(cfg.get_path("data.window_len", 32))
    cfg_h = float(cfg.get_path("data.label.healthy_ratio", 0.8))
    strategy = str(cfg.get_path("data.label.strategy", "piecewise"))
    data_dir = pathlib.Path(cfg.get_path("paths.data_dir", "artifacts/data"))
    index_path = data_dir / "index.json"
    if not index_path.exists():
        print(f"[error] {index_path} not found -- run 01_prepare_data.py first")
        return 2
    index = json.loads(index_path.read_text(encoding="utf-8"))

    # (display name, healthy_ratio or None, factory)
    labels: list[tuple[str, float | None, object]] = [("linear", None, linear_rul)]
    labels += [(f"piecewise_h{h:g}", h,
                (lambda n, h=h: piecewise_linear_rul(n, healthy_ratio=h))) for h in arts.h]

    print(f"config        : {pathlib.Path(arts.config).name}")
    print(f"window_len    : {L}   config healthy_ratio = {cfg_h}")
    print(f"label families: {', '.join(n for n, _, _ in labels)}")
    print()
    header = (f"{'bearing':>10} {'snaps':>6} {'life':>7} {'label':>14} "
              f"{'const_one':>10} {'best_const':>11} {'const_zero':>11} {'LI':>7}  check")
    print(header)
    print("-" * len(header))

    out: dict = {"config": str(arts.config), "window_len": L, "bearings": {}}
    failures = 0
    checked = 0
    per_label: dict[str, list[dict]] = {n: [] for n, _, _ in labels}

    for rec in index:
        if rec.get("protocol") != "single":
            continue
        with np.load(rec["file"]) as d:
            if "test/ratio" not in d.files:
                continue
            measured_ratio = np.asarray(d["test/ratio"], dtype=np.float64)
            bid = np.asarray(d["test/bearing_id"])
            ends = np.asarray(d["test/window_end"]).astype(int)
            life_arr = np.asarray(d["test/lifetime"], dtype=np.float64)

        for b in sorted(set(bid.tolist())):
            m = bid == b
            n_snap = int(ends[m].max()) + 1
            life = float(life_arr[m][0])
            end_idx = _scored_ends(n_snap, L)
            disp = f"{rec.get('name', '?')}::{b}"
            out["bearings"][disp] = {"n_snapshots": n_snap, "lifetime": life, "labels": {}}

            for label_name, h_val, fn in labels:
                ratio = fn(n_snap)[end_idx].astype(np.float64)
                a = _anchors(ratio, life)
                out["bearings"][disp]["labels"][label_name] = a
                per_label[label_name].append(a)

                # cross-check the row that reproduces the stored labels: the
                # linear row when the config is linear, else the piecewise row
                # whose h matches the config.
                is_target = (strategy == "linear" and h_val is None) or (
                    strategy == "piecewise" and h_val is not None and abs(h_val - cfg_h) < 1e-9
                )
                check = ""
                if is_target:
                    meas = float(np.sqrt(np.mean((1.0 - measured_ratio[m]) ** 2))) * life
                    rel = abs(meas - a["const_one"]) / a["const_one"] if a["const_one"] else float("nan")
                    ok = rel < arts.tol
                    failures += 0 if ok else 1
                    checked += 1
                    check = "OK" if ok else f"FAIL({rel * 100:.2f}%)"

                print(f"{disp:>10} {n_snap:>6} {life:>7.1f} {label_name:>14} "
                      f"{a['const_one']:>10.2f} {a['best_const']:>11.2f} "
                      f"{a['const_zero']:>11.2f} {a['label_informativeness']:>7.3f}  {check}")
            print()

    print("=== mean over bearings ===")
    print(f"{'label':>14} {'const_one':>10} {'best_const':>11} {'LI':>7}")
    for label_name, _, _ in labels:
        rows = per_label[label_name]
        if not rows:
            continue
        co = float(np.mean([r["const_one"] for r in rows]))
        bc = float(np.mean([r["best_const"] for r in rows]))
        li = float(np.mean([r["label_informativeness"] for r in rows]))
        print(f"{label_name:>14} {co:>10.2f} {bc:>11.2f} {li:>7.3f}")
        out.setdefault("means", {})[label_name] = {
            "const_one": co, "best_const": bc, "label_informativeness": li}

    if arts.out:
        pathlib.Path(arts.out).write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {arts.out}")

    print()
    if failures:
        print(f"[FAIL] {failures}/{checked} cross-check(s) mismatch the stored labels")
        return 1
    print(f"[PASS] {checked} '{strategy}' anchor(s) agree with the stored test labels")
    print("       read `best_const` (not `const_one`) as the hurdle; `LI` is the")
    print("       share of const_one error that no constant can explain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
