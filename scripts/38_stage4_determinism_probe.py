"""Stage 4 determinism probe: two layers, both must pass before any formal run.

Frozen protocol: ``audit/STAGE4_PROTOCOL_FREEZE.md`` R5 / red line 29.

Stage 5 exposed the evaluation-path non-determinism (finding B8). Stage 4 adds
THREE brand-new architectures, so we prove both layers again, separately:

Probe A -- TRAINING determinism
    Train the same ``(model, config, fold, seed)`` >= 3 times and compare,
    bit for bit:
      * checkpoint state_dict hash
      * training history (train_loss / val_loss / task / mmd curves)
      * validation trajectory (best_val_loss, best_epoch, epochs_run)

Probe B -- EVALUATION determinism
    Re-evaluate ONE stored checkpoint >= 3 times and compare, bit for bit:
      * RMSE (and every headline metric)
      * the prediction vector itself
      * the uncertainty outputs (predictive std / interval bounds)

Only when both probes PASS may ``scripts/36_run_stage4.py`` be used for the
formal run.

Usage
-----
    python scripts/38_stage4_determinism_probe.py --models resnet_bigru se_resnet_gru
    python scripts/38_stage4_determinism_probe.py --models resnet_bigru --repeats 3

Trains a handful of models (probe only) -- run AFTER code review + dry-run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

import numpy as np
from typing import Sequence

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from rulbench.config import Config  # noqa: E402
from rulbench.data.build import load_dataset  # noqa: E402
from rulbench.evaluation.inference import (  # noqa: E402
    clip_to_lifetime,
    predict_seed,
    ratio_to_minutes,
)
from rulbench.train.protocols import checkpoint_path, train_one  # noqa: E402
from rulbench.utils import resolve_device  # noqa: E402

BASE = REPO / "configs" / "base.yaml"
OUT = REPO / "artifacts" / "reports" / "stage4_probe"
PROBE_SEED = 42
PROBE_FOLD = 1


def _merge(*paths) -> Config:
    return Config.load(str(BASE), *[str(p) for p in paths])


def cfg_fold(k: int) -> Config:
    return _merge(REPO / "configs" / "matched_folds" / f"fold{k}.yaml").freeze()


def _sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------- Probe A
def probe_a(art, model: str, cfg: Config, device, repeats: int, work: pathlib.Path) -> dict:
    """Train the same (model, fold, seed) ``repeats`` times; compare bit for bit.

    Compares THREE things separately (freeze R5):
      * checkpoint file bytes (sha256) -- the weights themselves
      * training history (train/val loss curves) -- the optimisation trajectory
      * validation trajectory (best_val_loss / best_epoch / epochs_run) -- early
        stopping outcome, which is what actually selects the model
    """
    ckpts, hashes = [], []
    for i in range(repeats):
        out = work / f"probeA_{model}_rep{i}" / "checkpoints"
        out.mkdir(parents=True, exist_ok=True)
        ckpt = checkpoint_path(out, art["name"], model, PROBE_SEED)
        t0 = time.time()
        train_one(art, model, cfg, seed=PROBE_SEED, out_path=ckpt, device=device)
        ckpts.append(ckpt)
        hashes.append(_sha256_file(ckpt))
        print(f"  [A] {model} rep{i}: {time.time() - t0:6.1f}s  sha={hashes[-1][:16]}", flush=True)

    # compare the payloads (history + val trajectory + final weights) as well as
    # the file bytes
    import torch
    payloads = [torch.load(str(c), map_location="cpu", weights_only=False) for c in ckpts]
    keys = ["best_val_loss", "best_epoch", "epochs_run"]
    traj_ok = all(
        all(payloads[0].get(k) == payloads[j].get(k) for j in range(1, repeats))
        for k in keys
    )
    hist_ok = all(payloads[0].get("history") == payloads[j].get("history")
                  for j in range(1, repeats))

    # Explicit per-tensor comparison of the FINAL WEIGHTS. The file sha256 already
    # implies this, but stating it separately makes the claim checkable without
    # trusting the serialiser's byte layout (torch.save embeds metadata/ordering
    # that could in principle differ while the tensors are equal -- and vice versa).
    def _state(p):
        for cand in ("model_state", "state_dict", "model"):
            if cand in p and isinstance(p[cand], dict):
                return p[cand]
        return {k: v for k, v in p.items() if hasattr(v, "shape")}

    st0 = _state(payloads[0])
    n_tensors = len(st0)
    weights_ok, first_diff = True, None
    if n_tensors:
        for j in range(1, repeats):
            stj = _state(payloads[j])
            if set(st0) != set(stj):
                weights_ok, first_diff = False, f"key set differs (rep{j})"
                break
            for k in st0:
                a = np.ascontiguousarray(st0[k].detach().cpu().numpy())
                b = np.ascontiguousarray(stj[k].detach().cpu().numpy())
                if a.shape != b.shape or not np.array_equal(a, b):
                    weights_ok = False
                    first_diff = f"{k} differs (rep{j})"
                    break
            if not weights_ok:
                break

    hash_ok = len(set(hashes)) == 1
    return {
        "model": model, "repeats": repeats,
        "sha256": hashes,
        "n_tensors": n_tensors,
        "first_weight_difference": first_diff,
        "ckpt_hash_identical": hash_ok,
        "history_identical": bool(hist_ok),
        "val_trajectory_identical": bool(traj_ok),
        "final_weights_identical": bool(weights_ok),
        "pass": bool(hash_ok and hist_ok and traj_ok and weights_ok),
    }


# ------------------------------------------------------------------- Probe B
def _rmse_minutes(art, mean_ratio: np.ndarray, split: str = "test") -> float:
    """RMSE in MINUTES, computed exactly as ``report.py::_split_truth`` does.

    The split payload has no ``y`` field: truth is ``ratio * lifetime`` and the
    prediction is clipped to the bearing's own lifetime (audit B4).
    """
    payload = art["splits"][split]
    life = np.asarray(payload["lifetime"], float)
    truth_min = np.asarray(payload["ratio"], float) * life
    pred_min = clip_to_lifetime(ratio_to_minutes(mean_ratio, life), life)
    return float(np.sqrt(np.mean((truth_min - pred_min) ** 2)))


def probe_b(art, model: str, cfg: Config, run_dir: pathlib.Path, seeds: Sequence[int],
            device, repeats: int) -> dict:
    """Re-evaluate the SAME checkpoint set ``repeats`` times; compare bit for bit.

    Each repeat recomputes the full 5-seed ensemble from the stored checkpoints,
    so the reported RMSE is on the same caliber as the frozen tables.
    """
    preds, stds = [], []
    metrics = []
    for i in range(repeats):
        means = []
        for s in seeds:
            ck = checkpoint_path(run_dir, art["name"], model, int(s))
            res = predict_seed(art, model, cfg, ck, split="test", device=device, seed=int(s))
            means.append((np.asarray(res["mean"], dtype=np.float64),
                          np.asarray(res["std"], dtype=np.float64)))
        mean = np.mean(np.stack([m for m, _ in means], axis=0), axis=0)
        std = np.sqrt(np.mean(np.stack([s_ ** 2 for _, s_ in means], axis=0), axis=0))
        preds.append(mean)
        stds.append(std)
        metrics.append(_rmse_minutes(art, mean, "test"))
        print(f"  [B] {model} rep{i}: rmse={metrics[-1]:.6f}  "
              f"pred_hash={hashlib.sha256(mean.tobytes()).hexdigest()[:16]}", flush=True)

    pred_ok = all(np.array_equal(preds[0], p) for p in preds[1:])
    std_ok = all(np.array_equal(stds[0], s) for s in stds[1:])
    rmse_ok = len(set(metrics)) == 1
    return {
        "model": model, "repeats": repeats, "n_seeds": len(seeds),
        "rmse": metrics,
        "rmse_identical": rmse_ok,
        "prediction_vector_identical": bool(pred_ok),
        "uncertainty_identical": bool(std_ok),
        "pass": bool(rmse_ok and pred_ok and std_ok),
    }


def _ensure_seed_set(art, model: str, cfg: Config, device, run_dir: pathlib.Path,
                     seeds: Sequence[int]) -> None:
    """Train any missing seed checkpoint in ``run_dir`` (needed to form the ensemble).

    Probe A trains ONE seed on purpose (that is the minimal determinism test). Probe
    B, however, must evaluate the same 5-seed ensemble the frozen tables use, so the
    remaining seeds have to exist. Training them here is legitimate: Probe B is not
    about training, it only needs a fixed checkpoint set to re-evaluate.
    """
    for s in seeds:
        ck = checkpoint_path(run_dir, art["name"], model, int(s))
        if not ck.exists():
            t0 = time.time()
            train_one(art, model, cfg, seed=int(s), out_path=ck, device=device)
            print(f"  [B-setup] {model} seed{s}: {time.time() - t0:6.1f}s (trained)",
                  flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["resnet_bigru", "se_resnet_gru", "pure_bigru"],
                    help="new Stage 4 architectures to probe")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--fold", type=int, default=PROBE_FOLD)
    ap.add_argument("--probe", choices=["a", "b", "ab"], default="ab",
                    help="a = training determinism only; b = evaluation only; ab = both")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    device = resolve_device("cuda")
    cfg = cfg_fold(args.fold)
    data_dir = REPO / f"artifacts/data_fold{args.fold}"
    if not data_dir.exists():
        print(f"[error] {data_dir} not found -- run 01_prepare_data.py for the folds first")
        return 2

    art = load_dataset(sorted(data_dir.glob("single_cond*.npz"))[0])
    work = pathlib.Path(args.out)
    work.mkdir(parents=True, exist_ok=True)

    # Probe B re-evaluates the same 5-seed ensemble the frozen tables use, so its
    # RMSE is directly comparable to any stored ``results_*.csv`` row.
    configured = cfg.get_path("runtime.seeds") or [42, 123, 456, 789, 1011]
    seeds = [int(s) for s in configured]

    report = {"fold": args.fold, "repeats": args.repeats, "seeds": seeds,
              "probe": args.probe, "probe_a": [], "probe_b": []}
    for m in args.models:
        print(f"[probe] === {m} ===", flush=True)
        run_dir = work / f"probeA_{m}_rep{args.repeats - 1}" / "checkpoints"
        if args.probe in ("a", "ab"):
            a = probe_a(art, m, cfg, device, args.repeats, work)
            report["probe_a"].append(a)
            # Probe B re-evaluates the LAST checkpoint set Probe A produced. Under
            # Probe A all reps are bit-identical by construction, so any one works;
            # using the last keeps the two probes on the same artefact.
            run_dir = work / f"probeA_{m}_rep{args.repeats - 1}" / "checkpoints"
        if args.probe in ("b", "ab"):
            _ensure_seed_set(art, m, cfg, device, run_dir, seeds)
            b = probe_b(art, m, cfg, run_dir, seeds, device, args.repeats)
            report["probe_b"].append(b)

    results = report["probe_a"] + report["probe_b"]
    all_pass = all(r["pass"] for r in results) if results else False
    report["pass"] = all_pass
    (work / "stage4_probe.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    if report["probe_a"]:
        print("=== Probe A (training determinism) ===")
        for r in report["probe_a"]:
            print(f"  {r['model']:<16} hash={r['ckpt_hash_identical']} "
                  f"history={r['history_identical']} val_traj={r['val_trajectory_identical']} "
                  f"weights={r['final_weights_identical']} (n_tensors={r['n_tensors']}) "
                  f"-> {'PASS' if r['pass'] else 'FAIL'}")
            if r.get("first_weight_difference"):
                print(f"      !! first difference: {r['first_weight_difference']}")
    if report["probe_b"]:
        print("=== Probe B (evaluation determinism) ===")
        for r in report["probe_b"]:
            print(f"  {r['model']:<16} rmse={r['rmse_identical']} "
                  f"pred={r['prediction_vector_identical']} unc={r['uncertainty_identical']} "
                  f"-> {'PASS' if r['pass'] else 'FAIL'}")
    print()
    print(f"[probe] OVERALL: {'PASS' if all_pass else 'FAIL'}  (wrote {work / 'stage4_probe.json'})")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
