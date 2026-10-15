import pathlib
R = pathlib.Path(__file__).resolve().parents[1]
lines = []
for k in (1, 2):
    for arm in ("single", "cross"):
        for tag in ("vm_fold", "matched_fold"):
            name = f"{tag}{k}" + ("_b8" if tag == "matched_fold" else "")
            d = R / f"artifacts/reports/{name}/{arm}"
            csv = d / "results_uncorrected.csv"
            lines.append(f"{name}/{arm:<7} dir={d.is_dir()} csv={csv.exists()}")
# data dirs
for k in (1, 2):
    for p in ("artifacts/data_fold1", "artifacts/data_vm_fold1"):
        d = R / p
        lines.append(f"{p}: dir={d.is_dir()} npz={len(list(d.glob('*.npz'))) if d.is_dir() else 0}")
(R / "artifacts/reports/_tmp_dbg.txt").write_text("\n".join(lines), encoding="utf-8")
print("ok")
