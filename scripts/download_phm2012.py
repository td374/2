"""Download the IEEE PHM 2012 / PRONOSTIA dataset (the second benchmark).

    python scripts/download_phm2012.py                      # -> datasets/PHM2012
    python scripts/download_phm2012.py --dest D:/data/PHM2012
    python scripts/download_phm2012.py --method zip         # no git required
    python scripts/download_phm2012.py --verify-only        # just audit an existing copy

The original FEMTO-ST hosting page is gone; the community mirror below is the
one referenced by recent review papers. ``git`` is used when available because
a shallow clone is both smaller and resumable, otherwise the script falls back
to the repository ZIP archive.

After the transfer the script audits the result with
:func:`rulbench.data.raw_phm2012.verify` and writes ``download_report.json``
next to the data, so the S1 acceptance gate has machine-readable evidence of
what was actually downloaded.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

#: Community mirror referenced by e.g. the Wiley review of industrial
#: fault-diagnosis datasets. Upstream is FEMTO-ST, which no longer serves it.
MIRRORS = {
    "Lucky-Loek": {
        "url": "https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset",
        "note": "upstream community mirror (Learning_set incl. temperature)",
    },
    "wkzs111": {
        "url": "https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset",
        "note": "often-cited copy of the same tree",
    },
}
DEFAULT_MIRROR = "Lucky-Loek"
DEFAULT_BRANCH = "master"

#: Rough footprint, used only for the progress message.
EXPECTED_DOWNLOAD_MB = 490
EXPECTED_ON_DISK_MB = 3000


# --------------------------------------------------------------------- helpers
def log(msg: str) -> None:
    print(msg, flush=True)


def have_git() -> bool:
    return shutil.which("git") is not None


def dir_is_populated(path: pathlib.Path) -> bool:
    return path.exists() and any(path.iterdir())


def _human(mb: float) -> str:
    return f"{mb / 1024:.2f} GB" if mb >= 1024 else f"{mb:.0f} MB"


# ------------------------------------------------------------------- transfers
def clone_with_git(url: str, dest: pathlib.Path, branch: str = DEFAULT_BRANCH) -> None:
    """Shallow clone. Much smaller than a full history and resumable."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "git",
        "clone",
        "--depth",
        "1",
        "--branch",
        branch,
        "--progress",
        url,
        str(dest),
    ]
    log(f"[git] {' '.join(cmd)}")
    log(f"[git] about {_human(EXPECTED_DOWNLOAD_MB)} to transfer, "
        f"~{_human(EXPECTED_ON_DISK_MB)} once checked out")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"git clone failed with exit code {result.returncode}. "
            f"If the network blocks git, retry with --method zip."
        )


def download_with_zip(url: str, dest: pathlib.Path, branch: str = DEFAULT_BRANCH) -> None:
    """Fallback: fetch the repository ZIP archive and unpack it."""
    archive_url = f"{url.rstrip('/')}/archive/refs/heads/{branch}.zip"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        zip_path = tmp_path / "phm2012.zip"
        log(f"[zip] {archive_url}")

        started = time.time()
        with urllib.request.urlopen(archive_url) as response, zip_path.open("wb") as fh:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100.0 * done / total
                    print(f"\r[zip] {_human(done / 1e6)} / {_human(total / 1e6)} ({pct:5.1f}%)",
                          end="", flush=True)
            print()
        log(f"[zip] downloaded in {time.time() - started:.0f}s")

        log("[zip] unpacking ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_path / "x")
        roots = [p for p in (tmp_path / "x").iterdir() if p.is_dir()]
        if len(roots) != 1:
            raise RuntimeError(f"unexpected archive layout: {[p.name for p in roots]}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(roots[0]), str(dest))


def git_commit(path: pathlib.Path) -> str | None:
    if not (path / ".git").exists() or not have_git():
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except subprocess.CalledProcessError:  # pragma: no cover
        return None


# --------------------------------------------------------------------- auditing
def audit(dest: pathlib.Path) -> list[dict]:
    from rulbench.data.raw_phm2012 import CONDITION_META, lifetime, snapshot_count, verify

    rows = verify(dest)
    if not rows:
        log(f"[audit] no Bearing folders found under {dest}")
        return rows

    log("")
    log(f"{'bearing':>10} {'subset':>14} {'acc files':>10} {'life (min)':>11}  condition")
    for row in rows:
        meta = CONDITION_META[row["condition"]]
        log(
            f"{row['bearing']:>10} {str(row['subset']):>14} {row['n_acc_files']:>10} "
            f"{row['lifetime_minutes']:>11.1f}  "
            f"{meta['speed_rpm']:.0f} rpm / {meta['radial_load_n']:.0f} N"
        )

    by_condition: dict[int, list[str]] = {}
    for row in rows:
        by_condition.setdefault(row["condition"], []).append(str(row["bearing"]))
    log("")
    for condition in sorted(by_condition):
        bearings = by_condition[condition]
        log(f"[audit] condition {condition}: {len(bearings)} bearings -> {', '.join(bearings)}")

    total_snapshots = sum(int(snapshot_count(dest, r["condition"], int(str(r["bearing"]).split('_')[1]))) for r in rows)
    log(f"[audit] {len(rows)} bearings, {total_snapshots} recordings, "
        f"{sum(float(lifetime(dest, r['condition'], int(str(r['bearing']).split('_')[1]))) for r in rows) / 3600:.1f} h of data")

    expected = {1: 7, 2: 7, 3: 3}
    missing: list[str] = []
    for condition, count in expected.items():
        got = len(by_condition.get(condition, []))
        if got != count:
            missing.append(f"condition {condition}: expected {count}, found {got}")
    if missing:
        log("[audit] WARNING -- the download does not match the published inventory:")
        for item in missing:
            log(f"          {item}")
        log("          (Test_set is truncated by design; only use Full_Test_Set "
            "if you want the complete run-to-failure lives)")
    else:
        log("[audit] inventory matches the published challenge: 7 + 7 + 3 = 17 bearings")
    return rows


def write_report(dest: pathlib.Path, mirror: str, method: str, rows: list[dict]) -> pathlib.Path:
    report = {
        "dataset": "phm2012",
        "dest": str(dest),
        "mirror": MIRRORS[mirror]["url"],
        "method": method,
        "git_commit": git_commit(dest),
        "bearings": rows,
        "n_bearings": len(rows),
        "n_recordings": sum(int(r["n_acc_files"]) for r in rows),
        "total_minutes": round(sum(float(r["lifetime_minutes"]) for r in rows), 1),
    }
    path = dest / "download_report.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    log(f"[audit] wrote {path}")
    return path


# ------------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="download_phm2012",
        description="Download and audit the IEEE PHM 2012 / PRONOSTIA bearing dataset.",
    )
    parser.add_argument(
        "--dest", default=str(REPO_ROOT / "datasets" / "PHM2012"),
        help="target directory (default: <repo>/datasets/PHM2012)",
    )
    parser.add_argument(
        "--mirror", default=DEFAULT_MIRROR, choices=sorted(MIRRORS),
        help=f"which mirror to use (default: {DEFAULT_MIRROR})",
    )
    parser.add_argument(
        "--method", default="auto", choices=["auto", "git", "zip"],
        help="transfer method; 'auto' prefers git when installed",
    )
    parser.add_argument("--branch", default=DEFAULT_BRANCH, help="branch to fetch")
    parser.add_argument(
        "--force", action="store_true",
        help="proceed even if the target directory already holds files",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="skip the transfer and only audit what is already on disk",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dest = pathlib.Path(args.dest).expanduser()
    if not dest.is_absolute():
        dest = (REPO_ROOT / dest).resolve()

    log("=" * 72)
    log("IEEE PHM 2012 / PRONOSTIA downloader")
    log("=" * 72)
    log(f"mirror : {args.mirror} -- {MIRRORS[args.mirror]['note']}")
    log(f"source : {MIRRORS[args.mirror]['url']}")
    log(f"dest   : {dest}")

    if args.verify_only:
        if not dest.exists():
            log(f"[error] {dest} does not exist")
            return 2
        rows = audit(dest)
        if rows:
            write_report(dest, args.mirror, "verify-only", rows)
        return 0

    if dir_is_populated(dest) and not args.force:
        log("")
        log(f"[stop] {dest} already contains files.")
        log("       Nothing was touched. Re-run with --verify-only to audit it,")
        log("       or with --force to download into it anyway.")
        return 1

    method = args.method
    if method == "auto":
        method = "git" if have_git() else "zip"
        log(f"[auto] git {'found' if have_git() else 'not found'} -> using '{method}'")

    url = MIRRORS[args.mirror]["url"]
    try:
        if method == "git":
            clone_with_git(url, dest, args.branch)
        else:
            download_with_zip(url, dest, args.branch)
    except Exception as exc:  # noqa: BLE001 - surface a usable message
        log("")
        log(f"[error] {exc}")
        log("[hint] retry with:  python scripts/download_phm2012.py --method zip")
        return 2

    log("")
    rows = audit(dest)
    if rows:
        write_report(dest, args.mirror, method, rows)

    log("")
    log("=" * 72)
    log("next steps")
    log("=" * 72)
    log("  1. point configs/phm2012.yaml at this folder:")
    log(f"       data.root: {dest}")
    log("  2. python scripts/00_doctor.py --config configs/phm2012.yaml")
    log("  3. python scripts/01_prepare_data.py --config configs/phm2012.yaml")
    log("  4. check artifacts/phm2012/data/index.json against docs/dataset_selection.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
