"""Stage 2.1 -- label statistics + prediction-bias mechanism analysis.

**No training.** Everything here is computed from

* the stored label arrays inside the prepared datasets (``artifacts/data_*``), and
* predictions re-derived from the **existing** ``se_resnet_bigru`` checkpoints
  via ``predict_ensemble`` (a pure function of (checkpoint, data, seed) since the
  A10 seeding fix, so this reproduces the Stage-2 numbers bit-for-bit and does
  **not** retrain anything).

It answers the open question left by Stage 2: *why* does the model's advantage
over the deployable zero-learning baseline collapse as the plateau grows? Three
candidate mechanisms are separated explicitly:

* **A -- a signed drift in the prediction** (``bias`` moves away from 0 as h
  grows);
* **B -- collapse toward a constant** (bias ~ 0 but the error stops depending on
  the truth);
* **C -- stage-dependent error** (early / middle / late life disagree in sign).

Two reporting disciplines matter here and are easy to get wrong:

1. **Bias must be reported in normalised RUL units as well as minutes.** The
   three test bearings live 52 / 339 / 114 minutes, so "-66 min" means a
   different RUL fraction in each; the comparable quantity is
   ``pred_ratio - ratio``.
2. **Life-stage bins must use the normalised ratio**, not absolute minutes, for
   the same reason.

Every per-bearing quantity is aggregated with ``nanmean`` -- a bearing whose
life is short enough to have no windows in a stage would otherwise poison the
whole average.

Outputs ``artifacts/reports/stage2_bias.json`` (+ a scatter companion) and prints
one row per label point: ``Var(y)``, ``sd(y)``, ``N_decline/N``, ``Corr(y,t)``,
``R2(y~t)``, ``const_mean``, ``best_const``, model ``RMSE``, bias (min & ratio),
median bias, early/middle/late bias, ``G`` and ``Delta RMSE``.

    python scripts/26_stage2_bias.py
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
from rulbench.data.build import load_dataset  # noqa: E402
from rulbench.evaluation.inference import clip_to_lifetime, predict_ensemble, ratio_to_minutes  # noqa: E402
from rulbench.utils import resolve_device  # noqa: E402

MODEL = "se_resnet_bigru"

#: (display, h, report_dir)  -- data_dir is read from each run's resolved config
POINTS = [
    ("linear", 0.0, "artifacts/reports/s2_chain/linear"),
    ("h=0.5", 0.5, "artifacts/reports/s2_chain/h0.5"),
    ("h=0.6", 0.6, "artifacts/reports/s2_chain/h0.6"),
    ("h=0.7", 0.7, "artifacts/reports/s2_chain/h0.7"),
    ("h=0.8", 0.8, "artifacts/reports/s2_chain/h0.8"),
    ("h=0.9", 0.9, "artifacts/reports/s2_chain/h0.9"),
]

#: life-stage bins on the TRUE normalised RUL (1 = new, 0 = failure)
EARLY, LATE = 2.0 / 3.0, 1.0 / 3.0

AGG_KEYS = ("var_minutes", "sd_minutes", "var_ratio", "sd_ratio", "decline_share",
            "corr_y_t", "r2_y_t", "corr_r_u", "r2_r_u", "mean_ratio", "n_windows")


def _nanmean(values):
    arr = np.asarray(values, float)
    return float(np.nanmean(arr)) if arr.size and np.any(np.isfinite(arr)) else float("nan")


def label_stats(ratio: np.ndarray, life: np.ndarray, t: np.ndarray) -> dict:
    """Structure of the label for ONE bearing (lifetime is constant within it)."""
    ratio = np.asarray(ratio, float)
    life = np.asarray(life, float)
    t = np.asarray(t, float)
    y = ratio * life
    life0 = float(life[0]) if life.size else float("nan")
    u = t / life0  # normalised time in [0, 1]
    corr_y_t = float(np.corrcoef(y, t)[0, 1]) if np.std(t) > 0 else float("nan")
    corr_r_u = float(np.corrcoef(ratio, u)[0, 1]) if np.std(u) > 0 else float("nan")

    def _r2(x, z):
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
        resid = z - A @ coef
        ss = float(np.sum((z - z.mean()) ** 2))
        return float(1.0 - np.sum(resid**2) / ss) if ss > 0 else float("nan")

    return {
        "n_windows": int(y.size),
        "var_minutes": float(np.var(y)),
        "sd_minutes": float(np.std(y)),
        "var_ratio": float(np.var(ratio)),
        "sd_ratio": float(np.std(ratio)),
        "decline_share": float(np.mean(ratio < 1.0 - 1e-6)),
        "corr_y_t": corr_y_t,
        "r2_y_t": _r2(t, y),
        "corr_r_u": corr_r_u,
        "r2_r_u": _r2(u, ratio),
        "mean_ratio": float(ratio.mean()),
    }


def bias_stats(ratio: np.ndarray, life: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Signed-error summary for ONE bearing, in minutes AND normalised RUL.

    Also reports the ``pred_ratio ~ ratio`` regression, which separates the two
    ways a model can lose accuracy: a **shift** (non-zero intercept / bias) and
    an **attenuation** (slope < 1, i.e. the response stops tracking the truth).
    """
    ratio = np.asarray(ratio, float)
    life = np.asarray(life, float)
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    d_r = d / life
    pr = np.asarray(y_pred, float) / life
    slope = intercept = corr_pr = r2_pr = float("nan")
    if ratio.size > 2 and float(np.std(ratio)) > 0:
        slope, intercept = (float(v) for v in np.polyfit(ratio, pr, 1))
        corr_pr = float(np.corrcoef(ratio, pr)[0, 1])
        resid = pr - (slope * ratio + intercept)
        ss = float(np.sum((pr - pr.mean()) ** 2))
        r2_pr = float(1.0 - np.sum(resid**2) / ss) if ss > 0 else float("nan")
    out = {
        "bias_min": float(np.mean(d)),
        "median_bias_min": float(np.median(d)),
        "bias_ratio": float(np.mean(d_r)),
        "median_bias_ratio": float(np.median(d_r)),
        "share_over": float(np.mean(d > 0)),
        "rmse": float(np.sqrt(np.mean(d**2))),
        "rmse_ratio": float(np.sqrt(np.mean(d_r**2))),
        # Formal bias contribution: Bias^2 / MSE = mean(d)^2 / mean(d^2).
        # Bounded in [0, 1]; 0 => the error is pure variance, 1 => pure offset.
        "bias2_mse_min": float(np.mean(d) ** 2 / np.mean(d**2)) if np.mean(d**2) > 0 else float("nan"),
        "bias2_mse_ratio": float(np.mean(d_r) ** 2 / np.mean(d_r**2)) if np.mean(d_r**2) > 0 else float("nan"),
        # Prediction-response compression: how much of the TRUE spread the model
        # still reproduces. slope < 1 and disp < 1 both indicate compression.
        "sd_true_ratio": float(np.std(ratio)),
        "sd_pred_ratio": float(np.std(pr)),
        "dispersion_ratio": float(np.std(pr) / np.std(ratio)) if float(np.std(ratio)) > 0 else float("nan"),
        "slope_pr": slope,
        "intercept_pr": intercept,
        "corr_pr": corr_pr,
        "r2_pr": r2_pr,
        "n_windows": int(d.size),
    }
    for name, mask in (
        ("early", ratio >= EARLY),                       # healthy / near-new
        ("middle", (ratio < EARLY) & (ratio >= LATE)),
        ("late", ratio < LATE),                          # near failure
    ):
        key = f"_n_{name}"
        out[f"bias_{name}_min"] = float(np.mean(d[mask])) if np.any(mask) else float("nan")
        out[f"bias_{name}_ratio"] = float(np.mean(d_r[mask])) if np.any(mask) else float("nan")
        out[key] = int(mask.sum())
    return out


def run_point(display: str, h: float, report_rel: str, device) -> dict:
    report_dir = REPO / report_rel
    cfg = Config.load(report_dir / "resolved_config.yaml")
    data_dir = REPO / cfg.get_path("paths.data_dir")
    seeds = [int(s) for s in cfg.get_path("runtime.seeds", [42, 123, 456, 789, 1011])]
    run_dir = report_dir / "checkpoints"

    label_train, label_test, bias_per_bearing, scatter = [], [], {}, {}
    pool_ratio, pool_life, pool_yt, pool_yp = [], [], [], []
    for ds_path in sorted(data_dir.glob("single_cond*.npz")):
        art = load_dataset(ds_path)
        tr, te = art["splits"]["train"], art["splits"]["test"]
        tr_bid, te_bid = np.asarray(tr["bearing_id"]), np.asarray(te["bearing_id"])
        for b in sorted(set(tr_bid.tolist())):          # label stats per TRAIN bearing
            m = tr_bid == b
            label_train.append(label_stats(tr["ratio"][m], tr["lifetime"][m], tr["window_end"][m]))

        pred = predict_ensemble(art, MODEL, cfg, run_dir, seeds, split="test", device=device)
        life = np.asarray(te["lifetime"], float)
        ratio = np.asarray(te["ratio"], float)
        y_true = ratio * life
        # single-condition protocol => the correction factor is identically 1.0
        y_pred = clip_to_lifetime(ratio_to_minutes(pred["mean"], life), life)
        for b in sorted(set(te_bid.tolist())):
            m = te_bid == b
            key = f"{art['name']}::{b}"
            label_test.append(label_stats(ratio[m], life[m], te["window_end"][m]))
            bias_per_bearing[key] = bias_stats(ratio[m], life[m], y_true[m], y_pred[m])
            scatter[key] = {"ratio": ratio[m].tolist(), "y_true": y_true[m].tolist(),
                            "y_pred": y_pred[m].tolist()}
        pool_ratio.append(ratio)
        pool_life.append(life)
        pool_yt.append(y_true)
        pool_yp.append(y_pred)

    agg_label_train = {k: _nanmean([s[k] for s in label_train]) for k in AGG_KEYS}
    agg_label_test = {k: _nanmean([s[k] for s in label_test]) for k in AGG_KEYS}
    first = next(iter(bias_per_bearing.values()))
    bias_keys = [k for k in first if not k.startswith("_n_") and k != "n_windows"]
    bias_three = {k: _nanmean([s[k] for s in bias_per_bearing.values()]) for k in bias_keys}
    bias_three["n_windows"] = int(sum(s["n_windows"] for s in bias_per_bearing.values()))
    # Window-pooled caliber. ``Bias^2/MSE`` is a ratio of two means, so pooling the
    # windows is its natural aggregation; averaging the per-bearing ratios instead
    # lets the short-life bearing dominate (its RMSE is tiny, so its ratio is ~1).
    bias_pooled = bias_stats(np.concatenate(pool_ratio), np.concatenate(pool_life),
                             np.concatenate(pool_yt), np.concatenate(pool_yp))

    return {
        "label": display,
        "h": h,
        "label_train": agg_label_train,
        "label_test": agg_label_test,
        "bias_three_bearing": bias_three,
        "bias_pooled": bias_pooled,
        "per_bearing": bias_per_bearing,
        "label_train_per_bearing": label_train,
        "label_test_per_bearing": label_test,
        "scatter": scatter,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(REPO / "artifacts" / "reports" / "stage2_bias.json"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = resolve_device(args.device)

    chain = json.loads((REPO / "artifacts" / "reports" / "stage2_chain.json").read_text(encoding="utf-8"))
    chain_by_label = {r["label"]: r for r in chain["rows"]}

    rows = []
    hdr = (f"{'label':>7} {'caliber':>7} {'RMSE':>6} {'bias_r':>7} {'B2/MSE':>7} {'slope':>6} "
           f"{'sd_p':>6} {'sd_t':>6} {'disp':>6} | {'e_r':>6} {'m_r':>6} {'l_r':>6}")
    print("TRAIN label structure + error mechanism (*_r in RUL fraction)")
    print(hdr)
    print("-" * len(hdr))
    for display, h, rel in POINTS:
        res = run_point(display, h, rel, device)
        c = chain_by_label.get(display, {})
        res["chain_three_bearing"] = c.get("chain_three_bearing")
        res["chain_pooled"] = c.get("chain_pooled")
        rows.append(res)
        lt = res["label_train"]
        x = c.get("chain_three_bearing", {})
        for key, name in (("bias_three_bearing", "3-bear"), ("bias_pooled", "pooled")):
            b = res[key]
            rmse = x.get("rmse_model", float("nan")) if key == "bias_three_bearing" else \
                (c.get("chain_pooled", {}) or {}).get("rmse_model", float("nan"))
            tag = f"{display} {lt['decline_share'] * 100:.0f}%" if key == "bias_three_bearing" else ""
            print(f"{tag:>7} {name:>7} {rmse:>6.2f} {b['bias_ratio']:>+7.3f} {b['bias2_mse_ratio']:>7.3f} "
                  f"{b['slope_pr']:>6.3f} {b['sd_pred_ratio']:>6.3f} {b['sd_true_ratio']:>6.3f} "
                  f"{b['dispersion_ratio']:>6.3f} | {b['bias_early_ratio']:>+6.3f} "
                  f"{b['bias_middle_ratio']:>+6.3f} {b['bias_late_ratio']:>+6.3f}")

    pathlib.Path(args.json).write_text(
        json.dumps({"model": MODEL,
                    "rows": [{k: v for k, v in r.items() if k != "scatter"} for r in rows]},
                   indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"\nwrote {args.json}")

    scatter_path = pathlib.Path(args.json).with_name("stage2_bias_scatter.json")
    scatter_path.write_text(json.dumps({r["label"]: r["scatter"] for r in rows}, ensure_ascii=False),
                            encoding="utf-8")
    print(f"wrote {scatter_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
