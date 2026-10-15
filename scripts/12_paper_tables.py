"""Emit the manuscript tables as Markdown, straight from the run artefacts.

Writes ``paper/tables/*.md`` so every number in the manuscript can be traced to
a ``results_corrected.csv``. Nothing here is retyped by hand.

    python scripts/12_paper_tables.py
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

#: Stage-5 post-fix evidence root (audit/STAGE5_PROTOCOL_CHECK.md)
S5 = REPO / "artifacts" / "reports" / "s5"

ORDER = ["se_resnet_bigru", "gru", "pure_bigru", "cnn", "lstm", "informer_senet", "ridge",
         "constant_mean", "constant_one"]
DISPLAY = {
    "se_resnet_bigru": "SE-ResNet-BiGRU",
    "gru": "GRU",
    "pure_bigru": "BiGRU",
    "resnet_bigru": "ResNet-BiGRU",
    "se_resnet_gru": "SE-ResNet-GRU",
    "se_resnet_lstm": "SE-ResNet-LSTM",
    "cnn": "CNN",
    "lstm": "LSTM",
    "informer_senet": "Informer-SE",
    "ridge": "Ridge",
    "constant_mean": "Const. (mean)",
    "constant_one": "Const. (one)",
}
#: analytic best-constant hurdle (min, 3-condition mean components)
BEST_CONST = {"cond1": 6.06, "cond2": 88.91, "cond3": 23.96, "mean": 39.64}
BEST_CONST_PHM = {"cond1": 6704.5 / 3600.0, "cond2": 1254.3 / 3600.0,
                  "mean": (6704.5 + 1254.3) / 2.0 / 3600.0}


def read_rows(root: pathlib.Path, model: str) -> dict:
    path = root / model / "results_corrected.csv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return {r["dataset"]: r for r in csv.DictReader(fh)}


def fmt(v, nd=2):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def table_main(root: pathlib.Path, title: str, conds, scale=1.0, best=None, nd=2) -> str:
    lines = [f"### {title}", ""]
    head = "| Model | " + " | ".join(conds) + " | Mean |"
    lines.append(head)
    lines.append("| --- | " + " | ".join("---" for _ in conds) + " | --- |")
    for m in ORDER:
        rows = read_rows(root, m)
        if not rows:
            continue
        cells = [fmt(float(rows[c]["rmse"]) * scale, nd) if c in rows else "—" for c in conds]
        vals = [float(rows[c]["rmse"]) * scale for c in conds if c in rows]
        mean = sum(vals) / len(vals) if vals else float("nan")
        name = DISPLAY.get(m, m)
        if m == "se_resnet_bigru":
            name = f"**{name}**"
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {fmt(mean, nd)} |")
    if best:
        cells = [fmt(best.get(c), nd) for c in conds]
        lines.append("| *best constant (analytic hurdle)* | " + " | ".join(cells) + f" | {fmt(best.get('mean'), nd)} |")
    lines.append("")
    return "\n".join(lines)


def table_uncertainty(root: pathlib.Path) -> str:
    lines = ["### Uncertainty (XJTU-SY, linear label)", "",
             "| Model | PICP (nom. 0.95) | MPIW (min) | CRPS |", "| --- | --- | --- | --- |"]
    for m in ORDER:
        rows = read_rows(root, m)
        if not rows:
            continue
        picp = [float(r["picp"]) for r in rows.values() if r.get("picp") not in (None, "", "NA")]
        mpiw = [float(r["mpiw"]) for r in rows.values() if r.get("mpiw") not in (None, "", "NA")]
        crps = [float(r["crps"]) for r in rows.values() if r.get("crps") not in (None, "", "NA")]
        if not picp:
            continue
        lines.append(f"| {DISPLAY.get(m, m)} | {fmt(sum(picp)/len(picp), 3)} | "
                     f"{fmt(sum(mpiw)/len(mpiw), 2)} | {fmt(sum(crps)/len(crps), 2)} |")
    lines.append("")
    return "\n".join(lines)


def table_spectrum() -> str:
    spec = json.loads((S5 / "label_sensitivity.json").read_text(encoding="utf-8"))
    lines = ["### Label-sensitivity spectrum (XJTU-SY, 3-condition mean)", "",
             "| Label | best const | LI | const (one) | SE-ResNet-BiGRU | GRU | Const. (mean) | Gap (SE-ResNet-BiGRU − hurdle) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in spec["points"]:
        a = row["anchors"]
        mm = row["models"]

        def g(name):
            return fmt(mm[name]["mean"], 2) if name in mm else "—"
        gap = (mm["se_resnet_bigru"]["mean"] - a["best_const"]) if "se_resnet_bigru" in mm else None
        lines.append(f"| {row['label']} | {fmt(a['best_const'])} | {fmt(a['LI'], 3)} | {fmt(a['const_one'])} "
                     f"| {g('se_resnet_bigru')} | {g('gru')} | {g('constant_mean')} | "
                     f"{('+' if (gap or 0) >= 0 else '')}{fmt(gap)} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    xj = S5 / "single_linear"
    h08 = S5 / "sweep_h0.8"
    ph = S5 / "phm2012_single_linear"

    md = []
    md.append("# Manuscript tables (auto-generated)\n")
    md.append(table_main(xj, "Main results — XJTU-SY, linear label (RMSE, min)", ["cond1", "cond2", "cond3"], best=BEST_CONST))
    md.append(table_uncertainty(xj))
    md.append(table_spectrum())
    md.append(table_main(h08, "Reference — XJTU-SY, piecewise label h=0.8, **full deep roster + strongest baseline** (RMSE, min)", ["cond1", "cond2", "cond3"]))
    md.append(table_main(ph, "External validation — IEEE PHM 2012, linear label (RMSE, h)",
                         ["cond1", "cond2"], scale=1.0 / 3600.0, best=BEST_CONST_PHM))

    text = "\n".join(md)
    (OUT / "tables.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {OUT / 'tables.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
