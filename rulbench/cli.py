"""Command-line entry points.

Usage::

    python -m rulbench.cli prepare          --config configs/base.yaml
    python -m rulbench.cli single           --config configs/single.yaml
    python -m rulbench.cli cross            --config configs/cross.yaml
    python -m rulbench.cli cross            --config configs/cross_uda_off.yaml
    python -m rulbench.cli ablation         --config configs/ablation.yaml
    python -m rulbench.cli sweep            --config configs/base.yaml --sweep healthy_ratio

``--config`` files are always merged on top of ``configs/base.yaml`` (pass
``--no-base`` to opt out). ``--set a.b=c`` overrides individual values, so a
one-off variation never needs a new YAML file.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any, Dict, List, Sequence

from .config import Config, dump_config
from .utils import get_logger, resolve_device

log = get_logger()

DEFAULT_BASE = "configs/base.yaml"


# --------------------------------------------------------------------- helpers
def _coerce(text: str) -> Any:
    """Turn a CLI string into int / float / bool / None / str."""
    low = text.strip().lower()
    if low in {"true", "yes"}:
        return True
    if low in {"false", "no"}:
        return False
    if low in {"null", "none"}:
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    if "," in text:
        return [_coerce(part) for part in text.split(",")]
    return text


def build_config(args: argparse.Namespace) -> Config:
    paths: List[str] = []
    if not args.no_base:
        paths.append(args.base or DEFAULT_BASE)
    if args.config:
        paths.append(args.config)
    cfg = Config.load(*paths) if paths else Config()
    for item in args.set or []:
        key, _, value = item.partition("=")
        cfg = cfg.override(**{key.replace(".", "__"): _coerce(value)})
    return cfg.freeze()


def persist_config(cfg: Config, out_dir: str) -> None:
    dump_config(cfg, pathlib.Path(out_dir) / "resolved_config.yaml")


# ---------------------------------------------------------------- subcommands
def cmd_prepare(args: argparse.Namespace) -> int:
    from .pipelines.prepare import prepare

    cfg = build_config(args)
    out_dir = args.out or cfg.get_path("paths.data_dir", "artifacts/data")
    persist_config(cfg, out_dir)
    result = prepare(cfg, out_dir)
    log.info("prepared %d datasets", len(result["datasets"]))
    return 0


def cmd_single(args: argparse.Namespace) -> int:
    from .pipelines.run import run_single

    cfg = build_config(args)
    out = args.out or cfg.get_path("paths.report_dir", "artifacts/reports/single")
    persist_config(cfg, out)
    result = run_single(cfg, cfg.get_path("paths.data_dir"), out, device=resolve_device(cfg.get_path("runtime.device", "cuda")))
    _print_headline(result["headline"])
    return 0


def cmd_cross(args: argparse.Namespace) -> int:
    from .pipelines.run import run_cross

    cfg = build_config(args)
    out = args.out or cfg.get_path("paths.report_dir", "artifacts/reports/cross")
    persist_config(cfg, out)
    result = run_cross(cfg, cfg.get_path("paths.data_dir"), out, device=resolve_device(cfg.get_path("runtime.device", "cuda")))
    _print_headline(result["headline"])
    return 0


def cmd_ablation(args: argparse.Namespace) -> int:
    from .pipelines.run import run_ablation, run_ablation_cross

    cfg = build_config(args)
    out = args.out or cfg.get_path("paths.report_dir", "artifacts/reports/ablation")
    persist_config(cfg, out)
    single = run_ablation(
        cfg,
        cfg.get_path("paths.data_dir"),
        f"{out}/single",
        models=cfg.get_path("experiments.ablation_single.models"),
        device=resolve_device(cfg.get_path("runtime.device", "cuda")),
    )
    cross = run_ablation_cross(
        cfg,
        cfg.get_path("paths.data_dir"),
        f"{out}/cross",
        models=cfg.get_path("experiments.ablation_cross.models"),
        device=resolve_device(cfg.get_path("runtime.device", "cuda")),
    )
    _print_headline(single["headline"])
    _print_headline(cross["headline"])
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    from .sweep import run_sweep

    cfg = build_config(args)
    results = run_sweep(cfg, args.sweep, out_root=args.out)
    for name, table in results.items():
        log.info("sweep %s", name)
        _print_headline(table)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report which pieces of the environment are actually available."""
    cfg = build_config(args)
    checks: Dict[str, str] = {}
    try:
        import torch

        checks["torch"] = torch.__version__
        checks["cuda"] = str(torch.cuda.is_available())
    except ImportError as exc:  # pragma: no cover
        checks["torch"] = f"MISSING ({exc})"
    for module in ("numpy", "pandas", "scipy", "sklearn", "pywt", "yaml"):
        try:
            mod = __import__(module)
            checks[module] = getattr(mod, "__version__", "ok")
        except ImportError:
            checks[module] = "MISSING"

    from .data.registry import DATASETS, get_raw

    dataset = cfg.get_path("data.dataset", "xjtu_sy")
    checks["dataset"] = dataset
    checks["known_datasets"] = ", ".join(sorted(DATASETS))

    root = pathlib.Path(cfg.get_path("data.root", "."))
    checks["data_root"] = f"{root} exists={root.exists()}"
    if root.exists() and dataset in DATASETS:
        raw = get_raw(dataset)
        found = []
        for condition in cfg.get_path("prepare.conditions", [1, 2, 3]):
            present = raw.available_bearings(root, condition)
            found.append(f"c{condition}:[{','.join(str(b) for b in present)}]")
        checks["bearings_on_disk"] = " ".join(found) if found else "NONE"

    data_dir = pathlib.Path(cfg.get_path("paths.data_dir", "artifacts/data"))
    n_npz = len(list(data_dir.glob("*.npz"))) if data_dir.exists() else 0
    checks["prepared_datasets"] = f"{n_npz} npz files in {data_dir}"
    for key, value in checks.items():
        print(f"{key:>20}: {value}")
    return 0


def _print_headline(rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        print("(no results)")
        return
    metric = rows[0].get("metric", "rmse")
    print(f"{'model':<22} {'n':>3} {'mean':>10} {'std':>10}")
    for row in rows:
        print(f"{row['model']:<22} {row['n']:>3} {row['mean']:>10.4f} {row['std']:>10.4f}")


# -------------------------------------------------------------------- argparse
def add_common_arguments(parser: argparse.ArgumentParser, suppress_defaults: bool = False) -> None:
    """Register the options shared by every subcommand.

    They end up on both the root parser and each subparser so that
    ``rulbench --config x.yaml doctor`` and ``rulbench doctor --config x.yaml``
    both work. The second form is the one the ``scripts/*.py`` wrappers actually
    produce -- they prepend the subcommand name to ``sys.argv`` -- so without
    this the documented commands fail with "unrecognized arguments".

    On the subparser copies the defaults are suppressed; otherwise a subparser
    would reset a value the root parser had already parsed.
    """
    suppress = argparse.SUPPRESS
    parser.add_argument(
        "--config",
        default=suppress if suppress_defaults else None,
        help="override YAML merged on top of the base config",
    )
    parser.add_argument(
        "--base",
        default=suppress if suppress_defaults else DEFAULT_BASE,
        help=f"base config (default: {DEFAULT_BASE})",
    )
    parser.add_argument(
        "--no-base",
        action="store_true",
        default=suppress if suppress_defaults else False,
        help="do not auto-load the base config",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=suppress if suppress_defaults else None,
        metavar="KEY=VALUE",
        help="override one dotted key, repeatable",
    )
    parser.add_argument(
        "--out",
        default=suppress if suppress_defaults else None,
        help="output directory override",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rulbench",
        description="Rolling-bearing RUL benchmark (XJTU-SY + IEEE PHM 2012)",
    )
    add_common_arguments(parser)

    common = argparse.ArgumentParser(add_help=False)
    add_common_arguments(common, suppress_defaults=True)

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "prepare", parents=[common], help="build datasets from the configured raw root"
    ).set_defaults(func=cmd_prepare)
    sub.add_parser(
        "single", parents=[common], help="single-condition experiments"
    ).set_defaults(func=cmd_single)
    sub.add_parser(
        "cross", parents=[common], help="cross-condition transfer experiments"
    ).set_defaults(func=cmd_cross)
    sub.add_parser(
        "ablation", parents=[common], help="module ablations"
    ).set_defaults(func=cmd_ablation)
    sub.add_parser(
        "doctor", parents=[common], help="check environment and prepared data"
    ).set_defaults(func=cmd_doctor)

    sweep = sub.add_parser(
        "sweep", parents=[common], help="run a named parameter sweep"
    )
    sweep.add_argument("--sweep", required=True, help="key inside configs/sweep.yaml")
    sweep.set_defaults(func=cmd_sweep)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
