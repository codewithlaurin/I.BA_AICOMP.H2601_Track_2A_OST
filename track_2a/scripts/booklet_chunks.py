"""Page-oriented evidence chunks, independent of embeddings and storage."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    source_pdf: str
    parsed_sha256: str
    language: str | None
    page_number: int
    part: int
    text: str

    def payload(self) -> dict:
        return asdict(self)


def split_page(text: str, fits: Callable[[str], bool]) -> list[str]:
    """Keep pages whole when possible; split without losing non-whitespace characters.

    Prefer a paragraph boundary near the middle, then whitespace, then a character
    boundary. No overlap in this baseline. The caller counts actual model tokens,
    including its prefix and special tokens, rather than estimating from words.
    """
    if not text.strip():
        return []
    if fits(text):
        return [text]
    if len(text) < 2:
        raise ValueError("Token budget cannot fit one character and the model prefix")
    middle = len(text) // 2
    low, high = len(text) // 4, 3 * len(text) // 4
    for pattern in (r"\n\s*\n", r"\s+"):
        boundaries = [m.end() for m in re.finditer(pattern, text) if low < m.end() < high]
        if boundaries:
            middle = min(boundaries, key=lambda position: abs(position - middle))
            break
    left, right = text[:middle], text[middle:]
    # Discard only outer whitespace if it prevents a useful split.
    if not left.strip() or not right.strip():
        stripped = text.strip()
        if stripped != text:
            return split_page(stripped, fits)
    return split_page(left, fits) + split_page(right, fits)


def load_chunks(path: Path, fits: Callable[[str], bool]) -> list[Chunk]:
    """Read pages.json produced by parse_booklets.py; never infer missing language."""
    raw = path.read_bytes()
    data = json.loads(raw)
    if data.get("schema_version") != 1:
        raise ValueError(f"Unsupported pages.json schema: {path}")
    document_id = data["source_sha256"]
    if not isinstance(document_id, str) or not re.fullmatch(r"[0-9a-f]{64}", document_id):
        raise ValueError(f"Invalid source SHA-256: {path}")
    if not isinstance(data.get("source_pdf"), str) or not data["source_pdf"]:
        raise ValueError(f"Missing source PDF path: {path}")
    pages = data["pages"]
    if not isinstance(pages, list) or data.get("page_count") != len(pages):
        raise ValueError(f"Page count does not match pages: {path}")
    fingerprint = hashlib.sha256(raw).hexdigest()
    chunks = []
    seen = set()
    for page in pages:
        number = page["page_number"]
        if type(number) is not int or number < 1 or number in seen:
            raise ValueError(f"Invalid or duplicate page number in {path}: {number}")
        seen.add(number)
        text = page["markdown"]
        if not isinstance(text, str):
            raise ValueError(f"Page {number} must contain Markdown text: {path}")
        for part, passage in enumerate(split_page(text, fits), start=1):
            # Stable IDs let Qdrant upsert retries replace points instead of duplicating.
            chunk_id = str(uuid5(NAMESPACE_URL, f"booklet:{document_id}:{number}:{part}"))
            chunks.append(Chunk(chunk_id, document_id, data["source_pdf"], fingerprint,
                                data.get("language"), number, part, passage))
    if not chunks:
        raise ValueError(f"No nonempty page text in {path}; inspect the extraction first")
    return chunks
