"""Dataset assembly: raw data -> cached per-bearing features -> splits.

Two things make this module worth its length:

1. **Per-bearing feature caching.** Wavelet denoising plus 42 multi-domain
   descriptors on every snapshot is the most expensive non-GPU step in the
   pipeline. Cached per ``(condition, bearing, feature-config fingerprint)``,
   an iteration that only changes the model costs seconds instead of minutes.

2. **Explicit split provenance.** ``mrmr`` indices and the ``ChannelScaler`` are
   both fitted on the training split only and stored inside the artefact, so a
   reviewer (or a future you) can verify that no test information entered.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from ..config import Config
from ..utils import get_logger
from .features import extract_fft_amplitude, extract_multi_domain, select_mrmr, MULTI_DOMAIN_DIM
from .denoise import denoise_bearing
from .labels import build_labels
from .registry import get_raw, time_unit
from .scaling import ChannelScaler
from .windows import attach_lifetime, make_windows

log = get_logger()

#: default raw (unselected) feature width: 128 FFT + 42 multi-domain
RAW_FEATURE_DIM = 128 + MULTI_DOMAIN_DIM

#: number of accelerometer channels feeding the FFT block
N_CHANNELS = 2


# --------------------------------------------------------------- HI (plots)
def pca_health_indicator(features: np.ndarray) -> np.ndarray:
    """Monotonic 0->1 health indicator for visualisation only.

    Never used as a training target -- the supervised label is the piecewise
    linear RUL ratio. Kept here so the two concepts cannot be confused later.
    """
    from sklearn.decomposition import PCA

    features = np.asarray(features, dtype=np.float64)
    pc1 = PCA(n_components=1, random_state=0).fit_transform(features)[:, 0]
    hi = np.maximum.accumulate(pc1)
    span = hi.max() - hi.min()
    return ((hi - hi.min()) / span).astype(np.float32) if span > 1e-12 else np.zeros_like(hi, dtype=np.float32)


# ------------------------------------------------------------- feature cache
def _resolve_dataset(cfg: Config, dataset: str | None) -> str:
    """Explicit argument wins, then ``data.dataset``, then XJTU-SY."""
    return dataset or cfg.get_path("data.dataset", "xjtu_sy")


def fft_dim(cfg: Config) -> int:
    """Width of the FFT block: bins per channel x number of channels.

    Must be read from the config rather than hard-coded: the same
    ``data.fft.bins`` value spans a 12.8x wider bandwidth on PHM 2012 than on
    XJTU-SY, because the record is 2560 rather than 32768 samples long.
    """
    return N_CHANNELS * int(cfg.get_path("data.fft.bins", 64))


def _fingerprint(cfg: Config) -> str:
    """Short hash of everything that changes the raw feature matrix."""
    payload = {
        "denoise": cfg.get_path("data.denoise", {}),
        "fft": cfg.get_path("data.fft", {}),
        "multi_domain": cfg.get_path("data.multi_domain", {}),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(blob).hexdigest()[:10]


def bearing_features(
    root: str | pathlib.Path,
    condition: int,
    bearing_id: int,
    cfg: Config,
    cache_dir: str | pathlib.Path | None = None,
    use_cache: bool = True,
    dataset: str | None = None,
) -> Tuple[np.ndarray, int, float]:
    """Return ``(features (T, 170), n_snapshots, lifetime)`` for one bearing.

    ``n_snapshots`` is the length of the label axis; ``lifetime`` is the same
    span expressed in physical units of the dataset (minutes for XJTU-SY,
    seconds for PHM 2012). The two coincide on XJTU-SY -- one recording per
    minute -- but conflating them is exactly the bug that breaks a second
    dataset, so they are returned separately.
    """
    dataset = _resolve_dataset(cfg, dataset)
    raw = get_raw(dataset)

    cache_path = None
    if cache_dir is not None and use_cache:
        cache_path = (
            pathlib.Path(cache_dir)
            / f"{dataset}_c{condition}_b{bearing_id}_{_fingerprint(cfg)}.npz"
        )
        if cache_path.exists():
            with np.load(cache_path) as data:
                return (
                    data["features"].astype(np.float32),
                    int(data["n_snapshots"]),
                    float(data["lifetime"]),
                )

    n_snapshots = raw.snapshot_count(root, condition, bearing_id)
    life = float(raw.lifetime(root, condition, bearing_id))
    signal = raw.load_bearing_signal(root, condition, bearing_id)

    if cfg.get_path("data.denoise.enable", True):
        log.info("denoise %s c%d b%d (%d snapshots)", dataset, condition, bearing_id, signal.shape[0])
        signal = denoise_bearing(
            signal,
            wavelet=cfg.get_path("data.denoise.wavelet", "db4"),
            level=int(cfg.get_path("data.denoise.level", 3)),
            mode=cfg.get_path("data.denoise.mode", "soft"),
        )

    feats = np.concatenate(
        [
            extract_fft_amplitude(
                signal,
                bins=int(cfg.get_path("data.fft.bins", 64)),
                bin_start=int(cfg.get_path("data.fft.bin_start", 0)),
                log_scale=bool(cfg.get_path("data.fft.log_scale", False)),
            ),
            extract_multi_domain(
                signal,
                sample_rate=float(cfg.get_path("data.sample_rate", raw.SAMPLE_RATE)),
                source=cfg.get_path("data.multi_domain.source", "horizontal"),
            ),
        ],
        axis=1,
    ).astype(np.float32)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path, features=feats, n_snapshots=n_snapshots, lifetime=life
        )
    return feats, n_snapshots, life


# ------------------------------------------------------------------ splitting
def _windows_for_bearing(
    features_sel: np.ndarray,
    n_snapshots: int,
    lifetime: float,
    bearing_id: int,
    cfg: Config,
) -> Dict[str, np.ndarray]:
    window_len = int(cfg.get_path("data.window_len", 32))
    stride = int(cfg.get_path("data.stride", 1))
    labels = build_labels(
        n_snapshots,
        strategy=cfg.get_path("data.label.strategy", "piecewise"),
        healthy_ratio=float(cfg.get_path("data.label.healthy_ratio", 0.8)),
    )
    part = make_windows(features_sel, labels["ratio"], window_len=window_len, stride=stride)
    part["bearing_id"] = np.full(part["ratio"].shape[0], bearing_id, dtype=np.int32)
    part["condition"] = np.full(part["ratio"].shape[0], -1, dtype=np.int32)
    return attach_lifetime(part, lifetime)


def _merge(parts: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = list(parts[0].keys())
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}


def _select_features(
    feat_map: Dict[Tuple[int, int], np.ndarray],
    train_keys: Iterable[Tuple[int, int]],
    cfg: Config,
) -> Tuple[List[int], int]:
    """Fit mRMR on training bearings only; return (indices, final_dim)."""
    n_fft = fft_dim(cfg)
    if not cfg.get_path("data.mrmr.enable", True):
        width = next(iter(feat_map.values())).shape[1]
        return list(range(width)), width

    k = int(cfg.get_path("data.mrmr.k", 12))
    train_keys = list(train_keys)
    train_mat = np.concatenate([feat_map[key][:, n_fft:] for key in train_keys], axis=0)
    # relevance target = normalised position in life, the same proxy used to
    # build the labels, so no test information is required.
    targets = np.concatenate(
        [
            np.linspace(0.0, 1.0, feat_map[key].shape[0], dtype=np.float64)
            for key in train_keys
        ]
    )
    picked = select_mrmr(
        train_mat, targets, k=k, n_bins=int(cfg.get_path("data.mrmr.n_bins", 10))
    )
    indices = list(range(n_fft)) + [n_fft + i for i in picked]
    return indices, len(indices)


def _present_ids(root, raw, condition: int, ids: Sequence[int]) -> List[int]:
    """Configured ids that also exist on disk, preserving the configured order.

    Lets one split protocol be shared by datasets with different bearing
    budgets: PHM 2012 has 7 bearings under conditions 1-2 but only 3 under
    condition 3, so a list such as ``[1,2,3,4,5,6,7]`` degrades gracefully.
    """
    present = set(raw.available_bearings(root, condition))
    return [int(b) for b in ids if int(b) in present]


def _keep_present(root, raw, condition: int, ids: Sequence[int], what: str = "split") -> List[int]:
    """Same as :func:`_present_ids` but refuses to return an empty list."""
    kept = _present_ids(root, raw, condition, ids)
    if not kept:
        raise FileNotFoundError(
            f"{what}: none of bearings {[int(b) for b in ids]} exist for "
            f"condition {condition} under {root} "
            f"(present: {sorted(raw.available_bearings(root, condition))})"
        )
    return kept


def _train_val(
    root,
    raw,
    condition: int,
    train_ids: Sequence[int],
    val_ids: Sequence[int],
) -> Tuple[List[int], List[int]]:
    """Resolve a train/validation pair for one condition, degrading gracefully.

    A condition with fewer bearings than the protocol assumes (PHM 2012
    condition 3 has only 3) cannot satisfy ``val: [5]``. Rather than aborting,
    the last training bearing is held out as validation and a warning is
    logged, so the audit trail records that the split was adjusted.
    """
    train = _present_ids(root, raw, condition, train_ids)
    val = _present_ids(root, raw, condition, val_ids)
    if not train:
        raise FileNotFoundError(
            f"condition {condition} under {root}: none of the configured training "
            f"bearings {[int(b) for b in train_ids]} exist "
            f"(present: {sorted(raw.available_bearings(root, condition))})"
        )
    if not val:
        if len(train) < 2:
            raise FileNotFoundError(
                f"condition {condition} under {root}: need at least 2 bearings to "
                f"split train/validation, found {train}"
            )
        val = [train[-1]]
        train = train[:-1]
        log.warning(
            "condition %d: configured val ids %s absent; holding out bearing %d "
            "from training instead",
            condition,
            [int(b) for b in val_ids],
            val[0],
        )
    return train, val


# ---------------------------------------------------------------- public API
def build_single_condition(
    root: str | pathlib.Path,
    condition: int,
    cfg: Config,
    dataset: str | None = None,
) -> Dict[str, Any]:
    """Train on bearings 1-3, validate on 4, test on 5 -- within one condition."""
    dataset = _resolve_dataset(cfg, dataset)
    raw = get_raw(dataset)
    split_cfg = cfg.get_path("splits.single", {})
    train_ids, val_ids = _train_val(
        root,
        raw,
        condition,
        split_cfg.get("train", [1, 2, 3]),
        split_cfg.get("val", [4]),
    )
    test_ids = _keep_present(root, raw, condition, split_cfg.get("test", [5]), what="test")
    return _build(
        root=root,
        cfg=cfg,
        name=f"cond{condition}",
        groups={
            "train": [(condition, b) for b in train_ids],
            "val": [(condition, b) for b in val_ids],
            "test": [(condition, b) for b in test_ids],
        },
        meta={"protocol": "single", "condition": condition, "dataset": dataset},
        dataset=dataset,
    )


def build_cross_experiment(
    root: str | pathlib.Path,
    sources: Sequence[int],
    target: int,
    cfg: Config,
    dataset: str | None = None,
) -> Dict[str, Any]:
    """Source-condition(s) -> target-condition transfer task."""
    dataset = _resolve_dataset(cfg, dataset)
    raw = get_raw(dataset)
    split_cfg = cfg.get_path("splits.cross", {})
    src_train_ids = list(split_cfg.get("source_train", [1, 2, 3, 4]))
    src_val_ids = list(split_cfg.get("source_val", [5]))
    tgt_ids = list(split_cfg.get("target_test", [1, 2, 3, 4, 5]))

    src_train: List[Tuple[int, int]] = []
    src_val: List[Tuple[int, int]] = []
    for condition in sources:
        train, val = _train_val(root, raw, condition, src_train_ids, src_val_ids)
        src_train += [(condition, b) for b in train]
        src_val += [(condition, b) for b in val]
    tgt_test = [
        (target, b) for b in _keep_present(root, raw, target, tgt_ids, what="target_test")
    ]

    name = "+".join(str(s) for s in sources) + "->" + str(target)
    return _build(
        root=root,
        cfg=cfg,
        name=name,
        groups={"train": src_train, "val": src_val, "test": tgt_test},
        meta={
            "protocol": "cross",
            "dataset": dataset,
            "sources": list(sources),
            "target": target,
            "transductive_target": True,
        },
        dataset=dataset,
    )


def _build(
    root: str | pathlib.Path,
    cfg: Config,
    name: str,
    groups: Dict[str, List[Tuple[int, int]]],
    meta: Dict[str, Any],
    dataset: str | None = None,
) -> Dict[str, Any]:
    dataset = _resolve_dataset(cfg, dataset)
    cache_dir = cfg.get_path("data.cache_dir", None)
    use_cache = bool(cfg.get_path("data.use_cache", True))

    feat_map: Dict[Tuple[int, int], np.ndarray] = {}
    n_snapshots: Dict[Tuple[int, int], int] = {}
    lifetimes: Dict[Tuple[int, int], float] = {}
    for key in sorted({k for keys in groups.values() for k in keys}):
        feat_map[key], n_snapshots[key], lifetimes[key] = bearing_features(
            root, key[0], key[1], cfg,
            cache_dir=cache_dir, use_cache=use_cache, dataset=dataset,
        )

    indices, feat_dim = _select_features(feat_map, groups["train"], cfg)

    windows: Dict[str, Dict[str, np.ndarray]] = {}
    for split, keys in groups.items():
        parts = []
        for cond, bid in keys:
            part = _windows_for_bearing(
                feat_map[(cond, bid)][:, indices],
                n_snapshots[(cond, bid)],
                lifetimes[(cond, bid)],
                bid,
                cfg,
            )
            part["condition"] = np.full(part["ratio"].shape[0], cond, dtype=np.int32)
            parts.append(part)
        windows[split] = _merge(parts)

    # scaler: fit on the training split only, never on val/test
    scaler = ChannelScaler().fit(windows["train"]["X"])
    for split in windows:
        windows[split]["X"] = scaler.transform(windows[split]["X"])

    return {
        "name": name,
        "feature_dim": int(feat_dim),
        "window_len": int(cfg.get_path("data.window_len", 32)),
        "stride": int(cfg.get_path("data.stride", 1)),
        "splits": windows,
        "scaler": {"mean": scaler.mean_, "std": scaler.std_},
        "mrmr_indices": indices,
        "meta": {
            **meta,
            "groups": {k: [f"{c}_{b}" for c, b in v] for k, v in groups.items()},
            "lifetimes": {f"{c}_{b}": float(t) for (c, b), t in lifetimes.items()},
            "n_snapshots": {f"{c}_{b}": int(t) for (c, b), t in n_snapshots.items()},
            "n_windows": {k: int(v["ratio"].shape[0]) for k, v in windows.items()},
            "time_unit": time_unit(dataset),
        },
    }


# ------------------------------------------------------------------- storage
def save_dataset(artifact: Dict[str, Any], path: str | pathlib.Path) -> pathlib.Path:
    """Persist an artefact as a compressed npz plus a json sidecar."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, np.ndarray] = {
        "mrmr_indices": np.asarray(artifact["mrmr_indices"], dtype=np.int64),
        "scaler_mean": np.asarray(artifact["scaler"]["mean"]),
        "scaler_std": np.asarray(artifact["scaler"]["std"]),
    }
    for split, payload in artifact["splits"].items():
        for key, value in payload.items():
            arrays[f"{split}/{key}"] = np.asarray(value)
    np.savez_compressed(path, **arrays)
    with path.with_suffix(".json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "name": artifact["name"],
                "feature_dim": artifact["feature_dim"],
                "window_len": artifact["window_len"],
                "stride": artifact["stride"],
                "meta": artifact["meta"],
            },
            fh,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    return path


def load_dataset(path: str | pathlib.Path) -> Dict[str, Any]:
    """Inverse of :func:`save_dataset`; restores arrays, scaler and mRMR indices."""
    path = pathlib.Path(path)
    with path.with_suffix(".json").open("r", encoding="utf-8") as fh:
        meta = json.load(fh)

    splits: Dict[str, Dict[str, np.ndarray]] = {}
    with np.load(path) as data:
        for key in data.files:
            if "/" not in key:
                continue
            split, field = key.split("/", 1)
            splits.setdefault(split, {})[field] = data[key]
        scaler = {"mean": data["scaler_mean"], "std": data["scaler_std"]}
        mrmr_indices = data["mrmr_indices"].tolist()

    return {
        "name": meta["name"],
        "feature_dim": meta["feature_dim"],
        "window_len": meta["window_len"],
        "stride": meta["stride"],
        "meta": meta["meta"],
        "splits": splits,
        "scaler": scaler,
        "mrmr_indices": mrmr_indices,
    }


def dataset_fingerprint(artifact: Dict[str, Any]) -> str:
    """Stable id of a prepared dataset, used to invalidate stale model files."""
    blob = json.dumps(
        {
            "name": artifact["name"],
            "feature_dim": artifact["feature_dim"],
            "window_len": artifact["window_len"],
            "stride": artifact["stride"],
            "groups": artifact["meta"].get("groups"),
            "n_windows": artifact["meta"].get("n_windows"),
        },
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.md5(blob).hexdigest()[:10]


# ----------------------------------------------------------------- batch runs
def build_all(
    root: str | pathlib.Path,
    cfg: Config,
    out_dir: str | pathlib.Path,
    dataset: str | None = None,
) -> List[pathlib.Path]:
    """Prepare every dataset the experiment plan needs, in one call.

    Writes ``single_cond{1,2,3}.npz`` and ``cross_{task}.npz`` into ``out_dir``
    and returns the list of written paths. Because every dataset uses the same
    file names, each dataset must keep its own ``paths.data_dir`` (the PHM 2012
    config points at ``artifacts/phm2012/data``), which also keeps the
    ``single_cond*.npz`` / ``cross_*.npz`` globs in ``pipelines/run.py`` working
    unchanged.
    """
    dataset = _resolve_dataset(cfg, dataset)
    out_dir = pathlib.Path(out_dir)
    written: List[pathlib.Path] = []

    if cfg.get_path("prepare.single", True):
        for condition in cfg.get_path("prepare.conditions", [1, 2, 3]):
            artifact = build_single_condition(root, int(condition), cfg, dataset=dataset)
            written.append(save_dataset(artifact, out_dir / f"single_cond{condition}.npz"))
            log.info("built %s: %s", artifact["name"], artifact["meta"]["n_windows"])

    if cfg.get_path("prepare.cross", True):
        for sources, target in _cross_tasks(cfg):
            artifact = build_cross_experiment(root, sources, target, cfg, dataset=dataset)
            tag = _task_tag(sources, target)
            written.append(save_dataset(artifact, out_dir / f"cross_{tag}.npz"))
            log.info("built %s: %s", artifact["name"], artifact["meta"]["n_windows"])

    return written


def _cross_tasks(cfg: Config) -> List[Tuple[Tuple[int, ...], int]]:
    explicit = cfg.get_path("prepare.cross_tasks", None)
    if explicit:
        out = []
        for raw in explicit:
            sources_part, target_part = raw.split("->")
            sources = tuple(int(s) for s in sources_part.split("+"))
            out.append((sources, int(target_part)))
        return out
    conditions = [int(c) for c in cfg.get_path("prepare.conditions", [1, 2, 3])]
    tasks: List[Tuple[Tuple[int, ...], int]] = []
    for src in conditions:
        for tgt in conditions:
            if src != tgt:
                tasks.append(((src,), tgt))
    for a in conditions:
        for b in conditions:
            if b <= a:
                continue
            for tgt in conditions:
                if tgt not in (a, b):
                    tasks.append(((a, b), tgt))
    return tasks


def _task_tag(sources: Sequence[int], target: int) -> str:
    return "+".join(str(s) for s in sources) + "_to_" + str(target)
