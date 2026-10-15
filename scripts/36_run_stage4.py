"""Stage 4 driver: module ablations of the proposed architecture.

Frozen protocol: ``audit/STAGE4_PROTOCOL_FREEZE.md`` (red lines 28-33).

Everything lands in a **fresh** root ``artifacts/reports/s4/``. The invalid
pre-fix generation (``single_linear/``, ``sweep_h*/``) and the frozen Stage 5 set
(``s5/``, ``s2_chain/``, ``matched_fold*/``) are never written to (red line 28).

Statistical structure (frozen by the user, 2026-10-08)
------------------------------------------------------
::

    L1 component contribution   pure_bigru -> resnet_bigru      (extractor)
                                resnet_bigru -> se_resnet_bigru (SE)
                                se_resnet_gru -> se_resnet_bigru (bidirectional)
    L1 supplementary family cmp  cnn vs pure_bigru  (NOT part of the dG chain)
    L2 label x architecture      G_m(h), dG_arch(h), observed crossover interval
    L3 architecture x condition  se_resnet_bigru / resnet_bigru / pure_bigru

Blocks
------
``l1``    XJTU-SY single, linear label, the five L1 models
``l2``    XJTU-SY single, piecewise plateaus h in {0.5..0.9}, four L2 models
``l3``    XJTU-SY matched leave-one-bearing-out folds (single + cross arms)
``l3vm``  the same folds with training volume capped (reuses scripts/28-30)

``verify``  reuse hit-check: when a Stage 4 run shares a ``data_dir`` with a
            frozen Stage 5 run, ``run()`` skips training and only re-evaluates,
            so the stored metrics must agree **bit for bit**. A mismatch means
            the reuse premise is broken -> abort (red line 24).

Usage
-----
    python scripts/36_run_stage4.py run --blocks l1
    python scripts/36_run_stage4.py run --blocks l1,l2
    python scripts/36_run_stage4.py verify
"""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

# Import pandas BEFORE torch pulls in the CUDA runtime. On this machine
# constructing a DataFrame after a CUDA context exists segfaults (pandas 3.0.3 /
# torch 2.6.0+cu124); importing pandas up front is safe and makes the whole
# process immune. This is a defensive guard -- ``save_table`` no longer uses
# pandas at all (see rulbench/evaluation/report.py) -- kept so that any future
# pandas use inside the training loop cannot resurrect the crash.
try:  # pragma: no cover - optional dependency
    import pandas  # noqa: F401
except Exception:  # pragma: no cover
    pass

from rulbench.config import Config, dump_config  # noqa: E402
from rulbench.pipelines.run import run_cross, run_single  # noqa: E402
from rulbench.utils import resolve_device  # noqa: E402

BASE = REPO / "configs" / "base.yaml"
S4 = REPO / "artifacts" / "reports" / "s4"

# ------------------------------------------------------------------ rosters
#: L1 -- component contribution chain + the supplementary family comparison.
#: ``cnn`` is NOT part of the dG chain (different class: baselines.CNN1D), so the
#: analysis must report it separately (freeze doc section 3.1, red line 33).
L1_COMPONENT = ["pure_bigru", "resnet_bigru", "se_resnet_bigru", "se_resnet_gru"]
L1_FAMILY = ["cnn"]

#: L2 -- label x architecture. ``se_resnet_bigru`` reuses ``s5/sweep_h*``; the
#: other three must be trained at every h.
L2_MODELS = ["se_resnet_bigru", "resnet_bigru", "pure_bigru", "se_resnet_gru"]
H_VALUES = [0.5, 0.6, 0.7, 0.8, 0.9]

#: L3 -- architecture x cross-condition, THREE models only (red line 32).
L3_MODELS = ["se_resnet_bigru", "resnet_bigru", "pure_bigru"]
L3_FOLDS = [1, 2, 3, 4, 5]

#: B' (user decision 2026-10-11): the reduced L3. Cross arm keeps only the three
#: DUAL-SOURCE tasks -- "two known conditions -> one unknown target", which is
#: the question the paper actually asks. The six single-source tasks and the
#: volume-matched block are dropped: Batch 1 showed SE/ResNet/bidirectional
#: gains are all indistinguishable from zero, so a volume-matching arm would be
#: defending a confounding that no longer has an effect to explain.
L3_CROSS_TASKS = ["cross_1+2_to_3", "cross_1+3_to_2", "cross_2+3_to_1"]

#: columns that must agree exactly between two generations
METRIC_COLS = ["rmse", "mae", "nrmse", "phm_legacy", "phm_official",
               "picp", "mpiw", "crps", "scaling_factor"]


# --------------------------------------------------------------------- configs
def _merge(*paths: str | pathlib.Path) -> Config:
    return Config.load(str(BASE), *[str(p) for p in paths])


def cfg_xjtu_linear() -> Config:
    return _merge(REPO / "configs" / "linear.yaml").freeze()


def cfg_xjtu_h(h: float) -> Config:
    cfg = _merge(REPO / "configs" / "linear.yaml")
    return cfg.override(
        data__label__strategy="piecewise",
        data__label__healthy_ratio=float(h),
        paths__data_dir=f"artifacts/data_h{h}",
    ).freeze()


def cfg_fold(k: int) -> Config:
    """Matched leave-one-bearing-out fold ``k`` (same file Stage 3 used)."""
    return _merge(REPO / "configs" / "matched_folds" / f"fold{k}.yaml").freeze()


def cfg_fold_vm(k: int) -> Config:
    """Volume-matched variant of fold ``k`` (built by scripts/28)."""
    return _merge(REPO / "configs" / "matched_folds_vm" / f"fold{k}.yaml").freeze()


# --------------------------------------------------------------------- runners
def run_one(cfg: Config, data_dir: str, model: str, out_dir: pathlib.Path, device,
            cross: bool = False, datasets=None, merge: bool = False) -> None:
    """Run one (model, arm) unit.

    ``merge`` must be true whenever several models share ``out_dir`` -- i.e. the
    L3/L3vm arm directories. Under the supervised driver each model is its own
    subprocess, and a plain ``save_table`` overwrite would make the last writer
    truncate the other models' rows (B11). Per-model directories (L1/L2) keep
    ``merge=False`` so their CSVs stay exactly as before.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, out_dir / "resolved_config.yaml")
    t0 = time.time()
    runner = run_cross if cross else run_single
    kwargs = {"merge_tables": merge}
    if datasets is not None:
        kwargs["datasets"] = list(datasets)
    runner(cfg, data_dir, out_dir, models=[model], device=device, **kwargs)
    tag = "cross" if cross else "single"
    print(f"[s4] {out_dir.relative_to(REPO).as_posix():<52} {model:<18} {tag} {time.time() - t0:6.1f}s",
          flush=True)


def block_l3min(device) -> None:
    """L3 reduced (B'): single arm full, cross arm dual-source tasks only."""
    for k in L3_FOLDS:
        cfg = cfg_fold(k)
        dd = f"artifacts/data_fold{k}"
        for m in L3_MODELS:
            run_one(cfg, dd, m, S4 / f"l3_fold{k}" / "single", device, cross=False, merge=True)
            run_one(cfg, dd, m, S4 / f"l3_fold{k}" / "cross", device, cross=True,
                    datasets=L3_CROSS_TASKS, merge=True)


def block_l1(device) -> None:
    """L1 -- component contribution + supplementary family comparison."""
    cfg = cfg_xjtu_linear()
    dd = "artifacts/data_linear"
    for m in L1_COMPONENT + L1_FAMILY:
        run_one(cfg, dd, m, S4 / "l1_linear" / m, device)


def block_l2(device) -> None:
    """L2 -- label structure x architecture over the same h grid as Stage 5."""
    for h in H_VALUES:
        cfg = cfg_xjtu_h(h)
        dd = f"artifacts/data_h{h}"
        for m in L2_MODELS:
            run_one(cfg, dd, m, S4 / f"l2_h{h}" / m, device)


def block_l3(device) -> None:
    """L3 -- architecture x cross-condition, strictly paired folds."""
    for k in L3_FOLDS:
        cfg = cfg_fold(k)
        dd = f"artifacts/data_fold{k}"
        for m in L3_MODELS:
            run_one(cfg, dd, m, S4 / f"l3_fold{k}" / "single", device, cross=False, merge=True)
            run_one(cfg, dd, m, S4 / f"l3_fold{k}" / "cross", device, cross=True, merge=True)


def block_l3vm(device) -> None:
    """L3 volume-matched -- same folds with every arm capped to K = min."""
    for k in L3_FOLDS:
        cfg = cfg_fold_vm(k)
        dd = f"artifacts/data_vm_fold{k}"
        for m in L3_MODELS:
            run_one(cfg, dd, m, S4 / f"l3_vm_fold{k}" / "single", device, cross=False, merge=True)
            run_one(cfg, dd, m, S4 / f"l3_vm_fold{k}" / "cross", device, cross=True, merge=True)


def block_baselines(device) -> None:
    """Materialise the zero-learning floor (``const_mean`` / ``const_one``) in ``s4/``.

    No GPU work: both models are closed-form trivial predictors, so this only
    recomputes a constant from the same train split the deep models saw. Kept as
    its own block so the ``G`` denominator is always produced by the same script
    and the same data_dir as the numerator.
    """
    for sub, dd, cfg in _baseline_targets():
        for m in ("constant_mean", "constant_one"):
            run_one(cfg, dd, m, S4 / sub / m, device)


BLOCKS = {"l1": block_l1, "l2": block_l2, "l3": block_l3, "l3vm": block_l3vm,
          "l3min": block_l3min, "baselines": block_baselines}


# ---------------------------------------------------------------------- verify
def _read(path: pathlib.Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return {(r["dataset"], r["model"]): r for r in csv.DictReader(fh)}


def verify() -> int:
    """Models reused from a frozen Stage 5 run must reproduce it bit for bit.

    ``run()`` skips training when the checkpoint exists, so any model whose
    Stage 4 run shares ``data_dir`` with Stage 5 is trained once and evaluated
    again. That re-evaluation is a free A10 check: a mismatch means the reuse
    premise (same data + same config + post-fix checkpoints) is broken, and the
    Stage 4 numbers would be a hybrid -> abort (red line 24).
    """
    # (stage4 csv, stage5 csv) pairs that must agree
    pairs = [
        ("l1_linear", "single_linear"),
    ]
    checked = 0
    bad: list[tuple] = []
    for s4_name, s5_name in pairs:
        for m in ["se_resnet_bigru", "pure_bigru", "cnn"]:
            a = S4 / s4_name / m / "results_corrected.csv"
            b = REPO / "artifacts" / "reports" / "s5" / s5_name / m / "results_corrected.csv"
            ra, rb = _read(a), _read(b)
            if not ra or not rb:
                continue
            for key in sorted(set(ra) & set(rb)):
                for col in METRIC_COLS:
                    va, vb = ra[key].get(col, ""), rb[key].get(col, "")
                    if va == vb:
                        continue
                    try:
                        if abs(float(va) - float(vb)) <= 0:
                            continue
                    except (TypeError, ValueError):
                        pass
                    bad.append((s4_name, m, key, col, vb, va))
                checked += 1
    if checked == 0:
        print("[verify] no reusable Stage 4 output found yet -- run the 'l1' block first")
        return 0
    print(f"[verify] compared {checked} reused (dataset, model) rows x {len(METRIC_COLS)} columns")
    if bad:
        print(f"[verify] FAIL: {len(bad)} mismatch(es) vs the frozen Stage 5 set -- STOPPING")
        for s4n, m, (ds, mo), col, vs5, vs4 in bad[:40]:
            print(f"    {s4n:<14} {m:<16} {ds:<8} {col:<15} s5={vs5:<12} s4={vs4}")
        return 2
    print("[verify] PASS: reused models are bit-identical to the frozen Stage 5 set")
    return 0


# ------------------------------------------------------------------------ main
def _units(block: str):
    """Enumerate ``(block, label, callable)`` units so they can be run one at a time.

    Each unit is one ``run_one`` call: a single model on a single data_dir.
    Splitting the roster into units is what makes the driver resumable -- a hard
    crash (segfault, OOM kill, driver reset) takes down at most one unit, and
    re-launching skips every checkpoint that already exists.
    """
    if block == "l1":
        cfg = cfg_xjtu_linear()
        dd = "artifacts/data_linear"
        for m in L1_COMPONENT + L1_FAMILY:
            yield f"l1:{m}", (lambda m=m, cfg=cfg, dd=dd: run_one(cfg, dd, m, S4 / "l1_linear" / m, DEVICE))
    elif block == "l2":
        for h in H_VALUES:
            cfg = cfg_xjtu_h(h)
            dd = f"artifacts/data_h{h}"
            for m in L2_MODELS:
                yield f"l2:h{h}:{m}", (lambda h=h, m=m, cfg=cfg, dd=dd:
                                       run_one(cfg, dd, m, S4 / f"l2_h{h}" / m, DEVICE))
    elif block == "l3":
        for k in L3_FOLDS:
            cfg = cfg_fold(k)
            dd = f"artifacts/data_fold{k}"
            for m in L3_MODELS:
                yield f"l3:fold{k}:{m}:single", (lambda m=m, cfg=cfg, dd=dd, k=k:
                    run_one(cfg, dd, m, S4 / f"l3_fold{k}" / "single", DEVICE, cross=False, merge=True))
                yield f"l3:fold{k}:{m}:cross", (lambda m=m, cfg=cfg, dd=dd, k=k:
                    run_one(cfg, dd, m, S4 / f"l3_fold{k}" / "cross", DEVICE, cross=True, merge=True))
    elif block == "l3vm":
        for k in L3_FOLDS:
            cfg = cfg_fold_vm(k)
            dd = f"artifacts/data_vm_fold{k}"
            for m in L3_MODELS:
                yield f"l3vm:fold{k}:{m}:single", (lambda m=m, cfg=cfg, dd=dd, k=k:
                    run_one(cfg, dd, m, S4 / f"l3_vm_fold{k}" / "single", DEVICE, cross=False, merge=True))
                yield f"l3vm:fold{k}:{m}:cross", (lambda m=m, cfg=cfg, dd=dd, k=k:
                    run_one(cfg, dd, m, S4 / f"l3_vm_fold{k}" / "cross", DEVICE, cross=True, merge=True))
    elif block == "l3min":
        # B': full single arm + dual-source-only cross arm (3 models x 5 folds).
        # ``merge=True``: all three models share one arm directory, so each
        # per-model subprocess must accumulate into the CSV instead of
        # truncating the other two models' rows (B11).
        for k in L3_FOLDS:
            cfg = cfg_fold(k)
            dd = f"artifacts/data_fold{k}"
            for m in L3_MODELS:
                yield f"l3min:fold{k}:{m}:single", (lambda m=m, cfg=cfg, dd=dd, k=k:
                    run_one(cfg, dd, m, S4 / f"l3_fold{k}" / "single", DEVICE, cross=False, merge=True))
                yield f"l3min:fold{k}:{m}:cross", (lambda m=m, cfg=cfg, dd=dd, k=k:
                    run_one(cfg, dd, m, S4 / f"l3_fold{k}" / "cross", DEVICE, cross=True,
                            datasets=L3_CROSS_TASKS, merge=True))
    elif block == "baselines":
        # The zero-learning floor ``G_m = 1 - RMSE_m / RMSE_const_mean`` needs
        # ``const_mean`` *on the same data and the same label rule* as the deep
        # models. It is a trivial model: no training, no seeds, no GPU, so
        # materialising it inside ``s4/`` costs seconds and makes the analysis
        # root self-contained instead of reaching into the frozen ``s5/`` tree.
        # ``constant_one`` is the second mandatory trivial baseline (rule A1).
        for sub, dd, cfg in _baseline_targets():
            for m in ("constant_mean", "constant_one"):
                yield f"baselines:{sub}:{m}", (lambda sub=sub, dd=dd, cfg=cfg, m=m:
                                               run_one(cfg, dd, m, S4 / sub / m, DEVICE))
    else:
        raise SystemExit(f"unknown block {block!r}")


def _baseline_targets():
    """``(subdir, data_dir, cfg)`` for every report root that needs a floor.

    Must mirror exactly the data_dir + label rule used by the deep-model blocks
    or the ``G`` ratio compares two different labels.
    """
    yield "l1_linear", "artifacts/data_linear", cfg_xjtu_linear()
    for h in H_VALUES:
        yield f"l2_h{h}", f"artifacts/data_h{h}", cfg_xjtu_h(h)


#: set by ``main`` before any unit runs; the lambdas above close over it
DEVICE = None


def run_units(blocks, only: str | None = None) -> int:
    """Run every unit in ``blocks`` in THIS process (used for ``--one``)."""
    n = 0
    for b in blocks:
        for label, fn in _units(b):
            if only is not None and label != only:
                continue
            print(f"[s4] unit {label}", flush=True)
            fn()
            n += 1
    return n


def supervise(blocks) -> int:
    """Run every unit in its own subprocess so a hard crash cannot kill a block.

    A segfault kills the process with SIGSEGV (exit 139 on POSIX, STATUS_
    ACCESS_VIOLATION 0xC0000005 on Windows). With ``train_missing=True`` the
    checkpoints a unit already wrote survive, so re-running the same unit is
    how a crash is retried: the finished seeds are skipped and only the
    remaining ones train.
    """
    failures: list[tuple[str, int]] = []
    for b in blocks:
        for label, _ in _units(b):
            cmd = [sys.executable, str(pathlib.Path(__file__).resolve()), "run",
                   "--one", label, "--blocks", b]
            t0 = time.time()
            proc = subprocess.run(cmd, cwd=str(REPO))
            ok = proc.returncode == 0
            print(f"[s4] {label:<28} exit={proc.returncode:<4} {time.time() - t0:6.1f}s "
                  f"{'OK' if ok else 'FAILED'}", flush=True)
            if not ok:
                failures.append((label, proc.returncode))
    if failures:
        print(f"[s4] {len(failures)} unit(s) failed:", flush=True)
        for label, rc in failures:
            print(f"       {label} (exit {rc})", flush=True)
        return 1
    return 0


def main() -> int:
    global DEVICE
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "verify"])
    ap.add_argument("--blocks", default="l1",
                    help="comma-separated subset of: l1,l2,l3,l3vm")
    ap.add_argument("--one", default=None,
                    help="run exactly one unit in-process, e.g. l1:pure_bigru")
    ap.add_argument("--no-supervise", action="store_true",
                    help="run all units in one process (default: one subprocess per unit)")
    args = ap.parse_args()

    if args.mode == "verify":
        return verify()

    names = [b.strip() for b in args.blocks.split(",") if b.strip()]
    for name in names:
        if name not in BLOCKS:
            print(f"unknown block {name!r}; available: {sorted(BLOCKS)}")
            return 2

    DEVICE = resolve_device("cuda")

    if args.one is not None:
        n = run_units(names, only=args.one)
        if n == 0:
            print(f"[s4] no unit matched {args.one!r}")
            return 2
        return 0

    if args.no_supervise:
        t0 = time.time()
        for name in names:
            print(f"[s4] === block {name} ===", flush=True)
            BLOCKS[name](DEVICE)
        print(f"[s4] all blocks done in {time.time() - t0:.1f}s", flush=True)
        return verify()

    rc = supervise(names)
    if rc != 0:
        return rc
    return verify()


if __name__ == "__main__":
    sys.exit(main())
