"""Predictors behind the CLI: the Apertus inference and a neutral stub."""

from functools import lru_cache
import json
from pathlib import Path
import re
import sys
from typing import Protocol

from .contracts import Booklet, Request


class Predictor(Protocol):
    def predict(self, request: Request) -> dict:
        """Return a prediction satisfying contracts.validate_prediction.

        Raise on processing failure. The CLI will report it and exit nonzero.
        """
        ...


class NeutralPredictor:
    """Deterministic infrastructure stub, not semantic inference or a fallback."""

    def predict(self, request: Request) -> dict:
        return {
            "id": request.id,
            "label": 1,
            "label_name": "neutral",
            "evidence": [],
            "metrics": {
                "input_tokens": 0,
                "output_tokens": 0,
                "inference_time_ms": 0,
            },
        }


# pdfium marks hyphenation/ligature points with control characters inside words
# ("schweize\x02rische"); they break word matching and verbatim evidence.
_CONTROL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\ufeff]")


def clean_page_text(text: str) -> str:
    return _CONTROL.sub("", text)


def _shingles(text: str, k: int = 10) -> set:
    text = re.sub(r"\s+", " ", text.casefold())
    return {text[i:i + k] for i in range(max(len(text) - k + 1, 0))}


def choose_page_text(pdfium_text: str, pypdf_text: str) -> str:
    """pdfium is the primary extractor; pypdf replaces a page only when it holds text pdfium missed.

    pdfium skipped a paragraph inside a Form XObject on one booklet page (1 of 1,432 measured);
    pypdf had it. pypdf is not used by default because it inserts spurious spaces more often.
    """
    a, b = _shingles(pdfium_text), _shingles(pypdf_text)
    if b and len(pypdf_text) > len(pdfium_text) + 100 and len(b - a) / len(b) > 0.15:
        return pypdf_text
    return pdfium_text


# Soft hyphen (U+00AD) or pdfium's U+0002 marker at a line break: the word continues on the next line.
_HYPHEN_BREAK = re.compile("[ \t]*[\u00ad\x02][ \t]*\n[ \t]*|\n[ \t]*[\u00ad\x02][ \t]*\n?[ \t]*")
LABEL_COLUMN = 0.27  # booklet margin labels end left of 27 % of the page width
LABEL_GAP = 16.0     # points between consecutive lines of one margin label


def layout_text(page, textpage) -> str:
    """Reading-order text: margin labels placed before their paragraph, hyphenated line breaks joined.

    pdfium's own order lists the margin column ("Ausgangslage", "Forderungen der Initiative")
    after the whole page. Each line's first character box gives its column and height.
    """
    raw = textpage.get_text_range()
    if not raw.strip():
        return raw
    width = page.get_size()[0]
    labels, mains, offset = [], [], 0
    for line in raw.split("\r\n"):
        stripped = line.strip()
        if stripped:
            index = offset + line.index(stripped[0])
            try:
                left, _, _, top = textpage.get_charbox(index)
            except Exception:
                left, top = width, 0.0
            is_label = (left < width * LABEL_COLUMN and len(stripped) < 60
                        and not stripped.endswith((".", ",", ";", ":")))
            (labels if is_label else mains).append((top, left, stripped))
        offset += len(line) + 2
    mains.sort(key=lambda item: (-round(item[0]), item[1]))
    labels.sort(key=lambda item: -item[0])
    # Group label lines into blocks; a block is emitted before the first main line at or below it.
    blocks = []
    for top, _, text in labels:
        if blocks and blocks[-1][0] - top < LABEL_GAP * (len(blocks[-1][1]) + 1) and blocks[-1][2] - top < LABEL_GAP:
            blocks[-1][1].append(text); blocks[-1][2] = top
        else:
            blocks.append([top, [text], top])
    out, block_index = [], 0
    for top, _, text in mains:
        while block_index < len(blocks) and blocks[block_index][0] >= top - 2:
            out.append("\n" + " ".join(blocks[block_index][1]) + "\n")
            block_index += 1
        out.append(text + "\n")
    for block in blocks[block_index:]:
        out.append("\n" + " ".join(block[1]) + "\n")
    text = _HYPHEN_BREAK.sub("", "".join(out))
    text = text.replace("\u00ad", "-").replace("\x02", "")  # remaining soft hyphens are visible hyphens
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


@lru_cache(maxsize=16)
def pdf_pages(path: Path) -> tuple[dict, ...]:
    """Text per page, 1-based page numbers; empty pages skipped. CPU only, no OCR.

    get_text_range() rather than get_text_bounded(): the latter clips to the page
    box and lost whole pages on some booklets (34 of 1,400 pages measured).
    """
    import pypdfium2 as pdfium

    pages, reader = [], None
    with pdfium.PdfDocument(path) as document:
        for index in range(len(document)):
            page = document[index]
            try:
                textpage = page.get_textpage()
                try:
                    text = layout_text(page, textpage)
                finally:
                    textpage.close()
                # pdfium skipped text inside a Form XObject once; only such pages get the
                # slow pypdf second read (3 % of pages, ~0.1 s per booklet instead of ~2.7 s).
                has_form = any(obj.type == pdfium.raw.FPDF_PAGEOBJ_FORM for obj in page.get_objects())
            finally:
                page.close()
            if has_form:
                if reader is None:
                    from pypdf import PdfReader
                    reader = PdfReader(path)
                try:
                    text = choose_page_text(text, reader.pages[index].extract_text() or "")
                except Exception:
                    pass
            text = clean_page_text(text)
            if text.strip():
                pages.append({"text": text, "page": index + 1})
    if not pages:
        raise ValueError(f"no extractable text in {path}")
    return tuple(pages)


def vote_title(vote) -> str | None:
    if isinstance(vote, str):
        return vote
    if isinstance(vote, dict):
        return vote.get("title") or vote.get("name") or json.dumps(vote, ensure_ascii=False)
    return None if vote is None else str(vote)


class ApertusPredictor:
    """Task B: the reference is the only chunk. Task A: every booklet page is a chunk (full-document baseline).

    Never raises per case: any failure degrades to neutral with empty evidence and a line on
    stderr, so the CLI still writes a valid prediction for every id.
    """

    def predict(self, request: Request) -> dict:
        from .nli import classify, to_judge

        try:
            if isinstance(request.source, Booklet):
                chunks = list(pdf_pages(request.source.path))
            else:
                chunks = [{"text": request.source.text, "page": None}]
            pred = classify(request.claim.text, chunks, vote_title(request.vote))
            if pred["error"]:
                print(f"claimcheck: {request.id!r}: {pred['error']}", file=sys.stderr)
            return to_judge(pred, request.id)
        except Exception as exc:
            print(f"claimcheck: {request.id!r}: {exc.__class__.__name__}: {exc}; predicting neutral", file=sys.stderr)
            return NeutralPredictor().predict(request)
