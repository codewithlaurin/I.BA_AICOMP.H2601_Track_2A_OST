"""Side-by-side viewer: rendered PDF page (left) vs. the text our pipeline extracts (right).

    PYTHONPATH=src uv run python scripts/extraction_viewer.py BOOKLET.pdf --output output/viewer.html

Opens as a single HTML file (page images embedded). Use the page selector or the arrow keys.
"""

import argparse
import base64
import html
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claimcheck.predict import pdf_pages  # noqa: E402

TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Extraction check: {name}</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 0; background: #f4f4f4; }}
 header {{ padding: 8px 16px; background: #fff; border-bottom: 1px solid #ddd; display: flex; gap: 16px; align-items: center; }}
 main {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; padding: 12px; height: calc(100vh - 60px); box-sizing: border-box; }}
 .pane {{ background: #fff; border: 1px solid #ddd; overflow: auto; }}
 img {{ width: 100%; display: block; }}
 pre {{ white-space: pre-wrap; word-wrap: break-word; margin: 0; padding: 12px; font: 13px/1.4 ui-monospace, monospace; }}
 .meta {{ color: #666; font-size: 13px; }}
 .empty {{ color: #b00; }}
</style></head><body>
<header>
 <strong>{name}</strong>
 <label>Page <select id="sel"></select> of {count}</label>
 <button id="prev">&larr;</button><button id="next">&rarr;</button>
 <span class="meta" id="meta"></span>
</header>
<main>
 <div class="pane"><img id="img" alt="page"></div>
 <div class="pane"><pre id="txt"></pre></div>
</main>
<script>
 const pages = {pages};
 const sel = document.getElementById('sel');
 pages.forEach((p, i) => {{ const o = document.createElement('option'); o.value = i; o.textContent = p.page + (p.text.trim() ? '' : ' (no text)'); sel.appendChild(o); }});
 function show(i) {{
   const p = pages[i]; sel.value = i;
   document.getElementById('img').src = 'data:image/jpeg;base64,' + p.image;
   const pre = document.getElementById('txt'); pre.textContent = p.text || '(no text extracted)'; pre.className = p.text.trim() ? '' : 'empty';
   document.getElementById('meta').textContent = p.text.length + ' characters';
 }}
 sel.onchange = () => show(+sel.value);
 document.getElementById('prev').onclick = () => show(Math.max(0, +sel.value - 1));
 document.getElementById('next').onclick = () => show(Math.min(pages.length - 1, +sel.value + 1));
 document.onkeydown = e => {{ if (e.key === 'ArrowLeft') document.getElementById('prev').click(); if (e.key === 'ArrowRight') document.getElementById('next').click(); }};
 show(0);
</script></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/extraction_viewer.html"))
    parser.add_argument("--scale", type=float, default=1.3, help="render scale (1.0 = 72 dpi)")
    args = parser.parse_args()
    import pypdfium2 as pdfium

    text_by_page = {p["page"]: p["text"] for p in pdf_pages(args.pdf.resolve())}
    pages = []
    with pdfium.PdfDocument(args.pdf) as document:
        for index in range(len(document)):
            page = document[index]
            image = page.render(scale=args.scale).to_pil().convert("RGB")
            page.close()
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=70)
            pages.append({"page": index + 1, "text": text_by_page.get(index + 1, ""),
                          "image": base64.b64encode(buffer.getvalue()).decode("ascii")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(TEMPLATE.format(name=html.escape(args.pdf.name), count=len(pages),
                                           pages=json.dumps(pages)), encoding="utf-8")
    missing = [p["page"] for p in pages if not p["text"].strip()]
    print(f"wrote {args.output} ({len(pages)} pages; pages without text: {missing or 'none'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
