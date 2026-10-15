"""Render paper/manuscript.md to a self-contained, print-ready HTML preprint.

Figures are embedded as base64 data URIs so the single HTML file stands alone
(open it, or print it to PDF, with no external assets).

    python scripts/13_render_html.py
"""

from __future__ import annotations

import base64
import pathlib
import re
import sys

from markdown_it import MarkdownIt

REPO = pathlib.Path(__file__).resolve().parents[1]
PAPER = REPO / "paper"
FIGS = [
    ("fig1_label_design.png",
     "Figure 1. (a) Label definitions: the full-life linear label used here versus the "
     "piecewise plateau family. (b) Label informativeness LI falls monotonically as the "
     "plateau grows; LI is computable before any model is trained."),
    ("fig2_spectrum.png",
     "Figure 2. Headline result. Model and trivial-baseline RMSE (left axis) and analytic "
     "label informativeness (right axis) across the label spectrum. The gap between "
     "SE-ResNet-BiGRU and the optimal-constant hurdle is monotone in the plateau length and "
     "changes sign between h = 0.5 and h = 0.6: above that plateau no deep model tested beats "
     "a constant predictor."),
    ("fig3_main_results.png",
     "Figure 3. Per-condition main results. (a) XJTU-SY, linear label; (b) IEEE PHM 2012, "
     "linear label. Bars are measured five-seed ensembles; diamonds mark the analytic "
     "optimal-constant hurdle."),
]

CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; background: #f5f5f4; color: #1a1a1a;
  font: 15px/1.65 Georgia, "Times New Roman", serif; }
.page { max-width: 860px; margin: 0 auto; padding: 56px 62px 90px; background: #fff;
  box-shadow: 0 1px 26px rgba(0,0,0,.08); }
h1 { font-size: 25px; line-height: 1.3; margin: 0 0 .35em; letter-spacing: -.2px; }
h2 { font-size: 18px; margin: 1.9em 0 .5em; padding-bottom: .22em; border-bottom: 1px solid #e3e3e0; }
h3 { font-size: 15.5px; margin: 1.5em 0 .45em; }
p, li { font-size: 15px; }
strong { color: #111; }
a { color: #1f4e79; }
blockquote { margin: 1.2em 0; padding: .7em 1.1em; border-left: 3px solid #1f4e79;
  background: #f3f6fa; color: #26333f; font-size: 14.5px; }
blockquote p { margin: .35em 0; }
code { font-family: Consolas, "SF Mono", monospace; font-size: 13px;
  background: #f2f2ef; padding: 1px 5px; border-radius: 3px; }
pre { background: #f7f7f5; border: 1px solid #e6e6e2; border-radius: 6px;
  padding: 12px 14px; overflow-x: auto; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 1.1em 0; font-size: 13.5px;
  font-family: -apple-system, "Segoe UI", Arial, sans-serif; }
th, td { border-bottom: 1px solid #e6e6e2; padding: 7px 9px; text-align: left; }
th { background: #fafaf8; font-weight: 600; }
tbody tr:hover { background: #fbfbfa; }
hr { border: none; border-top: 1px solid #e6e6e2; margin: 2.2em 0; }
figure { margin: 1.8em 0; }
figure img { width: 100%; border: 1px solid #eaeae6; border-radius: 4px; background: #fff; }
figcaption { font-size: 13px; color: #4a4a4a; margin-top: .55em;
  font-family: -apple-system, "Segoe UI", Arial, sans-serif; }
.meta { font-size: 13.5px; color: #555; margin-bottom: 1.6em; }
@media print {
  body { background: #fff; } .page { box-shadow: none; max-width: none; padding: 0 8mm; }
  h2 { page-break-after: avoid; } figure, table { page-break-inside: avoid; }
}
"""


def figure_block() -> str:
    out = ["<h2>Figures</h2>"]
    for name, cap in FIGS:
        path = PAPER / "figures" / name
        if not path.exists():
            continue
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        out.append(
            f'<figure><img alt="{name}" src="data:image/png;base64,{b64}">'
            f"<figcaption>{cap}</figcaption></figure>"
        )
    return "\n".join(out)


def main() -> int:
    src = PAPER / "manuscript.md"
    text = src.read_text(encoding="utf-8")

    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    body = md.render(text)

    # The manuscript references Fig. 1-3; append the rendered figures right
    # after the Abstract's status note is not needed -- a Figures section at the
    # end keeps the flow readable.
    body += figure_block()

    html = (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>Label-sensitivity study — manuscript</title>\n"
        f"<style>{CSS}</style>\n</head>\n<body>\n<div class=\"page\">\n"
        f"{body}\n</div>\n</body>\n</html>\n"
    )
    out = PAPER / "manuscript.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({len(html) // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
