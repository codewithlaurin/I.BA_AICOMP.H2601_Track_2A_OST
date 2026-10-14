"""Predictors behind the CLI: the Apertus inference and a neutral stub."""

from functools import lru_cache
import os
import json
from pathlib import Path
import re
import sys
from typing import Protocol

from .contracts import Booklet, Request
from .cache import atomic_file, cache_dir, cache_lock, content_key, file_key


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


MIN_PAGE_CHARS = 50


def is_filler_page(text: str) -> bool:
    """Pages with nothing but a page number, a running header or a 'left blank' note carry no evidence.

    Examples: "17", "28 Aus produktionstechnischen Gründen leer.", "Page laissée vide ...".
    Measured by the remaining text after the page number, not by a phrase list, so all languages work.
    """
    body = re.sub(r"^\s*\d{1,3}\s*", "", text, count=1).strip()
    return len(body) < MIN_PAGE_CHARS


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
    labels, mains, hyphens, offset = [], [], [], 0
    for line in raw.split("\r\n"):
        stripped = line.strip()
        if stripped:
            index = offset + line.index(stripped[0])
            try:
                left, _, _, top = textpage.get_charbox(index)
            except Exception:
                left, top = width, 0.0
            if stripped in ("-", "\u00ad", "\x02"):
                # Some booklets emit the line-end hyphen as its own text run, listed
                # after the paragraph; its position says which line it belongs to.
                hyphens.append((top, left))
            else:
                is_label = (left < width * LABEL_COLUMN and len(stripped) < 60
                            and not stripped.endswith((".", ",", ";", ":")))
                (labels if is_label else mains).append((top, left, stripped))
        offset += len(line) + 2
    for hyphen_top, hyphen_left in hyphens:
        # The hyphen ends a line at the same height that starts left of it; when a
        # label and a paragraph line share the height, the nearer start wins.
        best = None
        for lines in (mains, labels):
            for i, (top, left, _) in enumerate(lines):
                if abs(top - hyphen_top) <= 6 and left < hyphen_left and (best is None or left > best[2]):
                    best = (lines, i, left)
        if best:
            lines, i, _ = best
            top, left, text = lines[i]
            lines[i] = (top, left, text + "\u00ad")
    mains.sort(key=lambda item: (-round(item[0]), item[1]))
    labels.sort(key=lambda item: -item[0])
    # Group label lines into blocks; a block is emitted before the first main line at or below it.
    blocks = []
    for top, _, text in labels:
        if blocks and blocks[-1][0] - top < LABEL_GAP * (len(blocks[-1][1]) + 1) and blocks[-1][2] - top < LABEL_GAP:
            blocks[-1][1].append(text); blocks[-1][2] = top
        else:
            blocks.append([top, [text], top])
    def label_text(block) -> str:
        return re.sub(r"[\u00ad\x02]\s*", "", " ".join(block[1]))

    out, block_index = [], 0
    for top, _, text in mains:
        # A label sits beside the first line of its paragraph; large titles start a
        # few points above the label, hence the tolerance.
        while block_index < len(blocks) and blocks[block_index][0] >= top - 14:
            out.append("\n" + label_text(blocks[block_index]) + "\n")
            block_index += 1
        out.append(text + "\n")
    for block in blocks[block_index:]:
        out.append("\n" + label_text(block) + "\n")
    text = _HYPHEN_BREAK.sub("", "".join(out))
    text = text.replace("\u00ad", "-").replace("\x02", "")  # remaining soft hyphens are visible hyphens
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


DOCUMENT_CACHE_VERSION = 1  # Bump when extraction, cleanup or filler-page rules change.


def pdf_pages(path: Path) -> tuple[dict, ...]:
    """Reuse extracted pages from disk; PDF content changes select a new cache entry."""
    path = Path(path).resolve()
    key = content_key({"pdf": file_key(path), "extraction": DOCUMENT_CACHE_VERSION})
    return _cached_pdf_pages(path, cache_dir() / "documents" / (key + ".json"))


@lru_cache(maxsize=16)
def _cached_pdf_pages(path: Path, destination: Path) -> tuple[dict, ...]:
    with cache_lock(destination):
        try:
            data = json.loads(destination.read_text(encoding="utf-8"))
            pages = data["pages"]
            if (not isinstance(pages, list) or not pages
                    or any(not isinstance(page, dict) or type(page.get("page")) is not int
                           or page["page"] < 1 or not isinstance(page.get("text"), str)
                           or not page["text"].strip() for page in pages)
                    or data["checksum"] != content_key(pages)):
                raise ValueError("invalid cached pages")
            return tuple(pages)
        except (OSError, ValueError, KeyError, TypeError):
            pages = _extract_pdf_pages(path)
            with atomic_file(destination) as temporary:
                temporary.write_text(json.dumps({"pages": pages, "checksum": content_key(pages)},
                                                ensure_ascii=False), encoding="utf-8")
            return pages


def _extract_pdf_pages(path: Path) -> tuple[dict, ...]:
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
            if not is_filler_page(text):
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
    """Task B: the reference is the only chunk. Task A: the booklet pages most similar to the
    claim (BM25 + FAISS over e5 embeddings, fused with RRF; see retrieval.py).
    Set CLAIMCHECK_FULL_DOCUMENT=1 to send
    every page instead (the baseline).

    Never raises per case: any failure degrades to neutral with empty evidence and a line on
    stderr, so the CLI still writes a valid prediction for every id.
    """

    def __init__(self, top_k: int | None = None, full_document: bool | None = None):
        from .retrieval import TOP_K

        self.top_k = top_k or int(os.environ.get("CLAIMCHECK_TOP_K", TOP_K))
        self.full_document = (os.environ.get("CLAIMCHECK_FULL_DOCUMENT") == "1"
                              if full_document is None else full_document)
        self._preparation_errors: dict[Path, str] = {}

    def prepare(self, requests: list[Request], *, strict: bool = False) -> None:
        """Cache each unique booklet before processing any claims; no LLM calls."""
        from .retrieval import booklet_index
        from .sparse import booklet_bm25

        paths = dict.fromkeys(request.source.path for request in requests if isinstance(request.source, Booklet))
        self._preparation_errors.clear()
        for number, path in enumerate(paths, 1):
            print(f"claimcheck: preparing booklet {number}/{len(paths)}: {path.name}", file=sys.stderr)
            try:
                pages = pdf_pages(path)
                if not self.full_document and len(pages) > self.top_k:
                    booklet_index(path, pages)
                    booklet_bm25(pages)
            except Exception as exc:
                self._preparation_errors[path] = f"{exc.__class__.__name__}: {exc}"
                print(f"claimcheck: cannot prepare {path}: {exc}", file=sys.stderr)
        if paths:
            print(f"claimcheck: prepared {len(paths) - len(self._preparation_errors)}/{len(paths)} booklets; "
                  "document cache ready", file=sys.stderr)
        if strict and self._preparation_errors:
            raise RuntimeError(f"failed to prepare {len(self._preparation_errors)} booklets")

    def predict(self, request: Request) -> dict:
        from .nli import MAX_EVIDENCE_CHARS, MAX_EVIDENCE_ITEMS, classify, to_judge

        try:
            retrieved: list[dict] = []
            if isinstance(request.source, Booklet):
                if request.source.path in self._preparation_errors:
                    raise RuntimeError(self._preparation_errors[request.source.path])
                pages = pdf_pages(request.source.path)
                if self.full_document:
                    chunks = list(pages)
                else:
                    from .retrieval import top_pages

                    retrieved = top_pages(request.source.path, pages, request.claim.text,
                                          vote_title(request.vote), self.top_k)
                    chunks = sorted(retrieved, key=lambda page: page["page"])  # reading order
            else:
                chunks = [{"text": request.source.text, "page": None}]
            pred = classify(request.claim.text, chunks, vote_title(request.vote))
            if pred["error"]:
                print(f"claimcheck: {request.id!r}: {pred['error']}", file=sys.stderr)
            result = to_judge(pred, request.id)
            if result["label"] != 1 and retrieved:
                # Hit@5 safety net: after the model's quotes, the retrieved pages themselves,
                # best first. One of the first five items has to overlap the gold passage.
                cited = {(item["page"], item["text"]) for item in result["evidence"]}
                for page in retrieved:
                    if len(result["evidence"]) >= MAX_EVIDENCE_ITEMS:
                        break
                    text = page["text"][:MAX_EVIDENCE_CHARS]
                    if (page["page"], text) not in cited and not any(
                            item["page"] == page["page"] and item["text"] in text for item in result["evidence"]
                            if len(item["text"]) >= len(text)):
                        result["evidence"].append({"page": page["page"], "text": text})
            return result
        except Exception as exc:
            print(f"claimcheck: {request.id!r}: {exc.__class__.__name__}: {exc}; predicting neutral", file=sys.stderr)
            return NeutralPredictor().predict(request)
