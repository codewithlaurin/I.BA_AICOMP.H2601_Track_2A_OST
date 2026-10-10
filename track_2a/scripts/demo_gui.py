"""Local demo GUI for the whole pipeline: booklet (or reference) + claim -> label + evidence.

    uv run --env-file .env python scripts/demo_gui.py [--port 8765] [--pdf-dir data/pdf]

Opens http://localhost:8765. Standard library only; the server runs ApertusPredictor exactly
as the judges' CLI does, so what you see here is what `python -m claimcheck` writes.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from claimcheck.contracts import Booklet, validate_request  # noqa: E402
from claimcheck.predict import ApertusPredictor, pdf_pages, vote_title  # noqa: E402
from claimcheck.retrieval import top_pages  # noqa: E402

LABELS = {0: "0 – Entailment", 1: "1 – Neutral", 2: "2 – Contradiction"}
COLORS = {0: "#1a7f37", 1: "#6e7781", 2: "#cf222e"}
PREDICTOR = ApertusPredictor()
PDFS: list[Path] = []

PAGE = """<!doctype html><meta charset="utf-8"><title>claimcheck demo</title>
<style>
body{{font:15px/1.45 system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#1f2328}}
form{{display:grid;grid-template-columns:1fr 1fr;gap:12px 24px}}
label{{display:block;font-weight:600;margin-bottom:4px}} textarea,input,select{{width:100%;box-sizing:border-box;padding:6px;font:inherit}}
.full{{grid-column:1/-1}} button{{padding:8px 20px;font:inherit;font-weight:600}}
.result{{margin-top:28px;border-top:1px solid #d0d7de;padding-top:16px}}
.label{{display:inline-block;color:#fff;padding:4px 12px;border-radius:6px;font-weight:700}}
.ev{{border-left:4px solid #d0d7de;padding:6px 12px;margin:10px 0;white-space:pre-wrap;background:#f6f8fa}}
.meta{{color:#57606a;font-size:13px}} details{{margin-top:12px}} summary{{cursor:pointer;font-weight:600}}
.err{{color:#cf222e;white-space:pre-wrap}}
</style>
<h1>claimcheck – Pipeline-Demo</h1>
<form method="post">
<div><label>Task</label><select name="task" onchange="document.getElementById('bk').hidden=this.value!='A';document.getElementById('rf').hidden=this.value!='B'">
<option value="A" {selA}>A – Booklet (PDF)</option><option value="B" {selB}>B – Reference passage</option></select></div>
<div><label>Vote (Titel der Vorlage, optional)</label><input name="vote" value="{vote}"></div>
<div id="bk" class="full" {hideA}><label>Booklet</label><select name="pdf">{pdf_options}</select></div>
<div id="rf" class="full" {hideB}><label>Reference passage</label><textarea name="reference" rows="5">{reference}</textarea></div>
<div class="full"><label>Claim</label><textarea name="claim" rows="3" required>{claim}</textarea></div>
<div><label>Claim language</label><select name="claim_lang">{lang_options}</select></div>
<div><label>Booklet/reference language</label><select name="src_lang">{src_lang_options}</select></div>
<div class="full"><button>Prüfen</button></div>
</form>
{result}
"""


def options(values, selected):
    return "".join(f'<option value="{html.escape(str(v))}" {"selected" if str(v) == str(selected) else ""}>'
                   f'{html.escape(str(v))}</option>' for v in values)


def render(form: dict, result: str = "") -> str:
    task = form.get("task", "A")
    return PAGE.format(
        selA="selected" if task == "A" else "", selB="selected" if task == "B" else "",
        hideA="" if task == "A" else "hidden", hideB="" if task == "B" else "hidden",
        vote=html.escape(form.get("vote", "")), claim=html.escape(form.get("claim", "")),
        reference=html.escape(form.get("reference", "")),
        pdf_options=options([p.name for p in PDFS], form.get("pdf", "")),
        lang_options=options(["de", "fr", "it"], form.get("claim_lang", "de")),
        src_lang_options=options(["de", "fr", "it"], form.get("src_lang", "de")),
        result=result,
    )


def run(form: dict) -> str:
    obj = {"id": "demo", "vote": form.get("vote") or None,
           "claim": {"text": form["claim"], "language": form.get("claim_lang", "de")}}
    if form.get("task") == "B":
        obj["reference"] = {"text": form["reference"], "language": form.get("src_lang", "de")}
    else:
        obj["booklet"] = {"path": form["pdf"], "language": form.get("src_lang", "de")}
    request = validate_request(obj, PDFS[0].parent if PDFS else ROOT)

    started = time.perf_counter()
    prediction = PREDICTOR.predict(request)
    total_ms = (time.perf_counter() - started) * 1000
    metrics = prediction["metrics"]

    retrieved_html = ""
    if isinstance(request.source, Booklet) and not PREDICTOR.full_document:
        pages = pdf_pages(request.source.path)
        hits = top_pages(request.source.path, pages, request.claim.text, vote_title(request.vote), PREDICTOR.top_k)
        rows = "".join(
            f'<div class="ev"><b>Seite {p["page"]}</b> <span class="meta">score {p["score"]:.3f}</span>\n'
            f'{html.escape(p["text"][:600])}{"…" if len(p["text"]) > 600 else ""}</div>' for p in hits)
        retrieved_html = (f'<details><summary>Abgerufene Seiten ({len(hits)} von {len(pages)}, '
                          f'bester zuerst)</summary>{rows}</details>')

    evidence = "".join(
        f'<div class="ev"><b>{"Seite " + str(e["page"]) if e["page"] else "Referenz"}</b>\n{html.escape(e["text"])}</div>'
        for e in prediction["evidence"]) or '<p class="meta">keine Evidenz (neutral)</p>'
    return f"""<div class="result">
<p><span class="label" style="background:{COLORS[prediction["label"]]}">{LABELS[prediction["label"]]}</span>
<span class="meta"> &nbsp; {metrics["input_tokens"]} input / {metrics["output_tokens"]} output tokens,
Modell {metrics["inference_time_ms"]} ms, gesamt {total_ms:.0f} ms (inkl. PDF + Retrieval)</span></p>
<h3>Evidenz ({len(prediction["evidence"])})</h3>{evidence}
{retrieved_html}
<details><summary>Rohe Vorhersage (Judge-Format)</summary><pre>{html.escape(json.dumps(prediction, ensure_ascii=False, indent=2))}</pre></details>
</div>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: str, status: int = 200):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send(render({}))

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        form = {k: v[0] for k, v in parse_qs(self.rfile.read(length).decode("utf-8")).items()}
        try:
            self._send(render(form, run(form)))
        except Exception as exc:  # show, never crash the server
            self._send(render(form, f'<p class="err">{html.escape(f"{exc.__class__.__name__}: {exc}")}</p>'))

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} {fmt % args}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--pdf-dir", type=Path, default=ROOT / "data" / "pdf")
    args = parser.parse_args()
    PDFS.extend(sorted(args.pdf_dir.glob("*.pdf")))
    if not PDFS:
        print(f"no PDFs in {args.pdf_dir}", file=sys.stderr)
    print(f"http://localhost:{args.port}  ({len(PDFS)} booklets from {args.pdf_dir})", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
