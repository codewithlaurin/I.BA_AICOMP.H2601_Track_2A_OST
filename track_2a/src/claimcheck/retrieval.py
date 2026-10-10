"""Page retrieval for Task A: e5 embeddings per page, one FAISS index per booklet.

Measured on 429 public rows (24 booklets): query = claim + vote finds the gold page in the
top 8 for 91 % of rows (BM25: 76 %). Everything runs on CPU; the embedding model must be
available locally in the judge container (set EMBEDDING_MODEL to a path, HF_HUB_OFFLINE=1).
"""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

DEFAULT_MODEL = "intfloat/multilingual-e5-small"
PASSAGE_CHARS = 2500   # e5-small reads 512 tokens; a booklet page is ~1,800 chars
TOP_K = 8


@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL), device="cpu")


def _normalize(vectors):
    import numpy as np

    vectors = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


_INDEXES: dict[str, object] = {}


def booklet_index(path: Path, pages: tuple):
    """FAISS inner-product index over the pages of one booklet (cosine on unit vectors), cached per file."""
    import faiss

    key = str(path)
    if key not in _INDEXES:
        texts = ["passage: " + page["text"][:PASSAGE_CHARS] for page in pages]
        vectors = _normalize(embedder().encode(texts, batch_size=16, show_progress_bar=False))
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        if len(_INDEXES) >= 16:
            _INDEXES.pop(next(iter(_INDEXES)))
        _INDEXES[key] = index
    return _INDEXES[key]


def top_pages(path: Path, pages: tuple, claim: str, vote: str | None, k: int = TOP_K) -> list[dict]:
    """Most relevant pages for the claim, best first, each with its retrieval score."""
    if len(pages) <= k:
        return [dict(page, score=None) for page in pages]
    index = booklet_index(path, pages)
    query = _normalize(embedder().encode(["query: " + " ".join(filter(None, [claim, vote]))]))
    scores, ids = index.search(query, k)
    return [dict(pages[i], score=float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]
