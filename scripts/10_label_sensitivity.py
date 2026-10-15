"""Assemble the label-sensitivity spectrum (IP4 / disclosure rule A6).

For each label definition (the linear endpoint plus piecewise plateaus
h in {0.5..0.9}) it pairs

* the **analytic** trivial-baseline ladder over the three single-condition test
  bearings (``const_one`` = always predict 1.0; ``best_const`` = optimal constant,
  the real hurdle; ``LI`` = label informativeness), and
* the **measured** RMSE of the models that were actually trained under that
  label (read from each run's ``summary_by_model.csv`` / ``results_corrected.csv``).

The headline quantity is ``gap = model_mean - best_const``: the plateau length at
which the proposed model stops losing to any constant predictor.

    python scripts/10_label_sensitivity.py
    python scripts/10_label_sensitivity.py --json artifacts/reports/label_sensitivity.json
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rulbench.config import Config  # noqa: E402
from rulbench.data.labels import linear_rul, piecewise_linear_rul  # noqa: E402

#: (display, label_strategy, healthy_ratio, data_dir)
POINTS = [
    ("linear", "linear", None, "artifacts/data_linear"),
    ("h=0.5", "piecewise", 0.5, "artifacts/data_h0.5"),
    ("h=0.6", "piecewise", 0.6, "artifacts/data_h0.6"),
    ("h=0.7", "piecewise", 0.7, "artifacts/data_h0.7"),
    ("h=0.8", "piecewise", 0.8, "artifacts/data_h0.8"),
    ("h=0.9", "piecewise", 0.9, "artifacts/data_h0.9"),
]
MODELS = ["se_resnet_bigru", "gru", "constant_mean", "constant_one"]


#: report dirs: linear lives under single_linear, plateaus under sweep_h{h}.
#: Both now live under the *post-fix* Stage-5 root (audit/STAGE5_PROTOCOL_CHECK.md).
S5_ROOT = REPO / "artifacts" / "reports" / "s5"


def report_dir(display: str, model: str) -> pathlib.Path:
    if display == "linear":
        return S5_ROOT / "single_linear" / model
    return S5_ROOT / f"sweep_h{display.split('=')[1]}" / model


def measured_summary(display: str, model: str) -> dict | None:
    path = report_dir(display, model) / "summary_by_model.csv"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        row = next(csv.DictReader(fh), None)
    return None if row is None else {"mean": float(row["mean"]), "std": float(row["std"])}


def measured_per_condition(display: str, model: str) -> dict:
    path = report_dir(display, model) / "results_corrected.csv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return {r["dataset"]: float(r["rmse"]) for r in csv.DictReader(fh)}


def analytic_anchors(data_dir: pathlib.Path, window_len: int) -> dict:
    """Windowed const_one / best_const / LI from the stored test splits."""
    index_path = data_dir / "index.json"
    if not index_path.exists():
        return {}
    index = json.loads(index_path.read_text(encoding="utf-8"))
    cat = {"const_one": [], "best_const": [], "const_zero": [], "LI": []}
    for rec in index:
        if rec.get("protocol") != "single":
            continue
        with np.load(rec["file"]) as d:
            if "test/ratio" not in d.files:
                continue
            ratio_full = np.asarray(d["test/ratio"], dtype=np.float64)
            bid = np.asarray(d["test/bearing_id"])
            ends = np.asarray(d["test/window_end"]).astype(int)
            life_arr = np.asarray(d["test/lifetime"], dtype=np.float64)
        for b in sorted(set(bid.tolist())):
            m = bid == b
            life = float(life_arr[m][0])
            r = ratio_full[m]
            one = life * float(np.sqrt(np.mean((r - 1.0) ** 2)))
            best = life * float(np.sqrt(np.mean((r - r.mean()) ** 2)))
            zero = life * float(np.sqrt(np.mean(r**2)))
            cat["const_one"].append(one)
            cat["best_const"].append(best)
            cat["const_zero"].append(zero)
            cat["LI"].append(1.0 - (best / one) ** 2 if one else float("nan"))
    return {k: float(np.mean(v)) for k, v in cat.items() if v}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "base.yaml"))
    ap.add_argument("--json", default=str(S5_ROOT / "label_sensitivity.json"))
    args = ap.parse_args()

    cfg = Config.load(args.config)
    L = int(cfg.get_path("data.window_len", 32))

    rows = []
    for display, _strat, _h, dd in POINTS:
        anch = analytic_anchors(REPO / dd, L)
        row = {"label": display, "anchors": anch, "models": {}}
        for model in MODELS:
            s = measured_summary(display, model)
            if s is None:
                continue
            row["models"][model] = {
                **s,
                "per_condition": measured_per_condition(display, model),
                "gap_vs_best_const": s["mean"] - anch.get("best_const", float("nan")),
            }
        rows.append(row)

    hdr = (f"{'label':>8} {'best_const':>11} {'LI':>6} {'const_one':>10} "
           f"{'proposed':>9} {'gru':>9} {'const_mean':>11} {'gap(prop)':>10}")
    print(f"window_len={L}")
    print(hdr)
    print("-" * len(hdr))
    for row in rows:
        a = row["anchors"]
        prop = row["models"].get("se_resnet_bigru", {}).get("mean")
        gru = row["models"].get("gru", {}).get("mean")
        cm = row["models"].get("constant_mean", {}).get("mean")
        # The hurdle is the analytic best_const: the optimum over ALL constant
        # predictors, hence a true lower bound for any constant baseline. The
        # measured constant_mean is always >= it (it uses the *training* label
        # mean, not the oracle test mean), so best_const is the stricter test.
        hurdle = a.get("best_const")
        gap = (prop - hurdle) if (prop is not None and hurdle is not None) else None
        fmt = lambda v: "     -" if v is None else f"{v:>9.2f}"  # noqa: E731
        print(f"{row['label']:>8} {a.get('best_const', float('nan')):>11.2f} "
              f"{a.get('LI', float('nan')):>6.3f} {a.get('const_one', float('nan')):>10.2f} "
              f"{fmt(prop)} {fmt(gru)} "
              f"{('        -' if cm is None else f'{cm:>11.2f}')} "
              f"{('         -' if gap is None else f'{gap:>10.2f}')}")

    pathlib.Path(args.json).write_text(
        json.dumps({"window_len": L, "points": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"\nwrote {args.json}")
    print("gap > 0 means the proposed model still loses to the strongest trivial baseline;")
    print("the sign flip across the label spectrum is the paper's headline result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
