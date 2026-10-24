"""Build report_T20.pdf from report/report_source.md.

Run from T20/:  .venv/bin/python tools/build_report.py
Writes report_T20.pdf at the repo root (the submission location).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import markdown
from weasyprint import HTML

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "report" / "report_source.md"
OUT = ROOT / "report_T20.pdf"

CSS = """
@page { size: A4; margin: 2cm 1.8cm; @bottom-center { content: counter(page); font-size: 9pt; color: #666; } }
body { font-family: "DejaVu Sans", sans-serif; font-size: 10.5pt; line-height: 1.45; color: #111; }
h1 { font-size: 17pt; border-bottom: 2px solid #222; padding-bottom: 4px; }
h2 { font-size: 13pt; margin-top: 22px; border-bottom: 1px solid #bbb; padding-bottom: 2px; }
h3 { font-size: 11.5pt; margin-top: 16px; }
table { border-collapse: collapse; margin: 10px 0; font-size: 9.5pt; width: 100%; }
th, td { border: 1px solid #999; padding: 4px 8px; text-align: left; }
th { background: #eee; }
code, pre { font-family: "DejaVu Sans Mono", monospace; font-size: 9pt; }
pre { background: #f6f6f6; border: 1px solid #ddd; padding: 8px; overflow-x: hidden; white-space: pre-wrap; }
blockquote { color: #555; border-left: 3px solid #ccc; margin-left: 0; padding-left: 10px; }
img { max-width: 100%; }
"""


def main() -> None:
    md_text = SRC.read_text(encoding="utf-8")
    html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    html = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html_body}</body></html>"
    HTML(string=html, base_url=str(ROOT)).write_pdf(OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
