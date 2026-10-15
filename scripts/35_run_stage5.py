"""Stage 5 driver: regenerate the formal evidence set under the post-fix protocol.

Everything lands in a **fresh** root ``artifacts/reports/s5/`` so the invalid
pre-fix generation (``single_linear/``, ``sweep_h*/``, ``phm2012/single_linear/``)
is never touched and its stale checkpoints can never be silently reused
(``run()`` skips training when a checkpoint exists -- see
``audit/STAGE5_PROTOCOL_CHECK.md`` §4.2, red line 24).

Blocks
------
``main``  §6.2  XJTU-SY single, linear label, 9 models (incl. ``pure_bigru`` = BiGRU)
``h``     §6.3  XJTU-SY single, piecewise plateaus h in {0.5..0.9}, 3 models
``phm``   §6.4  IEEE PHM 2012 single, linear label, 4 models

``verify``  determinism hit-check: the S5 proposed model must reproduce
            ``artifacts/reports/s2_chain/linear`` **bit for bit** on the stored
            metric columns. Any mismatch aborts before tables/figures are built
            (the user's explicit stop condition).

    python scripts/35_run_stage5.py run --blocks main,h,phm
    python scripts/35_run_stage5.py verify
"""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from rulbench.config import Config, dump_config  # noqa: E402
from rulbench.pipelines.run import run_single  # noqa: E402
from rulbench.utils import resolve_device  # noqa: E402

BASE = REPO / "configs" / "base.yaml"
S5 = REPO / "artifacts" / "reports" / "s5"

NEURAL_MAIN = ["se_resnet_bigru", "cnn", "lstm", "gru", "informer_senet", "pure_bigru"]
TRIVIAL_MAIN = ["constant_mean", "constant_one", "ridge"]
H_VALUES = [0.5, 0.6, 0.7, 0.8, 0.9]
H_MODELS = ["se_resnet_bigru", "gru", "constant_mean"]
#: §6.1 claims "no deep model beats the baselines" at the widely used h = 0.8, so the
#: *full* deep roster must be evaluated at that one point -- the spectrum block only
#: trains the proposed model + one deep baseline per h.
H08_MODELS = NEURAL_MAIN + ["constant_mean"]
PHM_MODELS = ["se_resnet_bigru", "constant_mean", "constant_one", "ridge"]

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


def cfg_phm_linear() -> Config:
    cfg = _merge(REPO / "configs" / "phm2012.yaml")
    return cfg.override(
        data__label__strategy="linear",
        paths__data_dir="artifacts/phm2012/data_linear",
    ).freeze()


# --------------------------------------------------------------------- runners
def run_one(cfg: Config, data_dir: str, model: str, out_dir: pathlib.Path, device) -> None:
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, out_dir / "resolved_config.yaml")
    t0 = time.time()
    run_single(cfg, data_dir, out_dir, models=[model], device=device)
    print(f"[s5] {out_dir.relative_to(REPO).as_posix():<52} {model:<18} {time.time() - t0:6.1f}s",
          flush=True)


def block_main(device) -> None:
    cfg = cfg_xjtu_linear()
    dd = "artifacts/data_linear"
    for m in NEURAL_MAIN + TRIVIAL_MAIN:
        run_one(cfg, dd, m, S5 / "single_linear" / m, device)


def block_h(device) -> None:
    for h in H_VALUES:
        cfg = cfg_xjtu_h(h)
        dd = f"artifacts/data_h{h}"
        for m in H_MODELS:
            run_one(cfg, dd, m, S5 / f"sweep_h{h}" / m, device)


def block_h08full(device) -> None:
    """Complete the deep roster at h = 0.8 so §6.1's claim rests on the S5 set.

    ``se_resnet_bigru`` / ``gru`` already have checkpoints here, so ``run()`` skips
    their training and only re-evaluates -- which doubles as an A10 check inside S5.
    """
    cfg = cfg_xjtu_h(0.8)
    dd = "artifacts/data_h0.8"
    for m in H08_MODELS:
        run_one(cfg, dd, m, S5 / "sweep_h0.8" / m, device)


def block_phm(device) -> None:
    cfg = cfg_phm_linear()
    dd = "artifacts/phm2012/data_linear"
    for m in PHM_MODELS:
        run_one(cfg, dd, m, S5 / "phm2012_single_linear" / m, device)


BLOCKS = {"main": block_main, "h": block_h, "h08full": block_h08full, "phm": block_phm}


# ---------------------------------------------------------------------- verify
def _read(path: pathlib.Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return {(r["dataset"], r["model"]): r for r in csv.DictReader(fh)}


def verify() -> int:
    """S5 proposed must reproduce s2_chain/linear bit for bit."""
    ref_dir = REPO / "artifacts" / "reports" / "s2_chain" / "linear"
    got_dir = S5 / "single_linear"
    ref, got = _read(ref_dir / "results_corrected.csv"), _read(got_dir / "se_resnet_bigru" / "results_corrected.csv")

    if not got:
        print("[verify] FAIL: no S5 proposed results found -- run the 'main' block first")
        return 2
    keys = sorted(set(ref) & set(got))
    if not keys:
        print("[verify] FAIL: no overlapping (dataset, model) rows")
        return 2

    bad = []
    for key in keys:
        for col in METRIC_COLS:
            a, b = ref[key].get(col, ""), got[key].get(col, "")
            if a != b:
                try:
                    if abs(float(a) - float(b)) <= 0:
                        continue
                except (TypeError, ValueError):
                    pass
                bad.append((key, col, a, b))

    print(f"[verify] compared {len(keys)} (dataset, model) rows x {len(METRIC_COLS)} columns")
    if bad:
        print(f"[verify] FAIL: {len(bad)} mismatching value(s) vs s2_chain -- STOPPING")
        for (ds, m), col, a, b in bad[:40]:
            print(f"    {ds:<8} {m:<20} {col:<15} s2_chain={a:<12} s5={b}")
        return 2
    print("[verify] PASS: S5 proposed is bit-identical to s2_chain (determinism closed)")
    return 0


# ------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "verify"])
    ap.add_argument("--blocks", default="main,h,phm",
                    help="comma-separated subset of: main,h,phm")
    args = ap.parse_args()

    if args.mode == "verify":
        return verify()

    device = resolve_device("cuda")
    names = [b.strip() for b in args.blocks.split(",") if b.strip()]
    t0 = time.time()
    for name in names:
        if name not in BLOCKS:
            print(f"unknown block {name!r}; available: {sorted(BLOCKS)}")
            return 2
        print(f"[s5] === block {name} ===", flush=True)
        BLOCKS[name](device)
    print(f"[s5] all blocks done in {time.time() - t0:.1f}s", flush=True)

    rc = verify()
    if rc != 0:
        print("[s5] determinism hit-check FAILED -- do NOT build tables/figures")
    return rc


if __name__ == "__main__":
    sys.exit(main())
