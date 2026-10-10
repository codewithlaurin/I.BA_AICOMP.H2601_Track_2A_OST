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


@lru_cache(maxsize=16)
def pdf_pages(path: Path) -> tuple[dict, ...]:
    """Text per page, 1-based page numbers; empty pages skipped. CPU only, no OCR.

    get_text_range() rather than get_text_bounded(): the latter clips to the page
    box and lost whole pages on some booklets (34 of 1,400 pages measured).
    """
    import pypdfium2 as pdfium

    pages = []
    with pdfium.PdfDocument(path) as document:
        for index in range(len(document)):
            page = document[index]
            try:
                textpage = page.get_textpage()
                try:
                    text = clean_page_text(textpage.get_text_range())
                finally:
                    textpage.close()
            finally:
                page.close()
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
