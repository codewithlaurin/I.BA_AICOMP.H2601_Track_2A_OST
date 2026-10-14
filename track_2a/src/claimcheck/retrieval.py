"""Page retrieval for Task A: e5 embeddings per page, one FAISS index per booklet.

Measured on 429 public rows (24 booklets): query = claim + vote finds the gold page in the
top 8 for 91 % of rows (BM25: 76 %). Everything runs on CPU; the embedding model must be
available locally in the judge container (set EMBEDDING_MODEL to a path, HF_HUB_OFFLINE=1).
"""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

from .cache import atomic_file, cache_dir, cache_lock, content_key

DEFAULT_MODEL = "intfloat/multilingual-e5-small"
PASSAGE_CHARS = 2500   # e5-small reads 512 tokens; a booklet page is ~1,800 chars
TOP_K = 8
INDEX_VERSION = 1


def _model_config() -> dict:
    model = os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL)
    config = {"model": model, "revision": os.environ.get("EMBEDDING_MODEL_REVISION")}
    local = Path(model).expanduser()
    if local.is_dir():
        config["model"] = str(local.resolve())
        # Rebuild if local weights, tokenizer or configuration files are replaced.
        config["files"] = [(str(file.relative_to(local)), file.stat().st_size, file.stat().st_mtime_ns)
                           for file in sorted(local.rglob("*")) if file.is_file()]
    return config


@lru_cache(maxsize=1)
def _load_embedder(model: str, revision: str | None, fingerprint: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model, revision=revision, device="cpu")


def embedder():
    config = _model_config()
    return _load_embedder(config["model"], config["revision"], content_key(config))


def _normalize(vectors):
    import numpy as np

    vectors = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


_INDEXES: dict[str, object] = {}


def booklet_index(path: Path, pages: tuple):
    """Reuse a persisted FAISS cosine index, also after eviction or process restart.

    The key includes full page text and page order, model/revision and encoding settings.
    Identical documents at different paths share an index; claims and top-k do not affect it.
    """
    digest = content_key({"version": INDEX_VERSION, "model": _model_config(),
                          "passage_chars": PASSAGE_CHARS, "pages": pages})
    destination = cache_dir() / "indexes" / (digest + ".faiss")
    key = str(destination)
    if key not in _INDEXES:
        # Load the embedding library (including PyTorch/sklearn) before FAISS,
        # even on cache hits: their OpenMP runtimes otherwise conflict on macOS.
        # Loading the model weights is deferred until embedding is necessary.
        import sentence_transformers  # noqa: F401
        import faiss

        with cache_lock(destination):
            try:
                index = faiss.read_index(str(destination))
                if index.ntotal != len(pages) or index.metric_type != faiss.METRIC_INNER_PRODUCT:
                    raise ValueError("cached index does not match the document")
            except (RuntimeError, OSError, ValueError):
                texts = ["passage: " + page["text"][:PASSAGE_CHARS] for page in pages]
                vectors = _normalize(embedder().encode(texts, batch_size=16, show_progress_bar=False))
                index = faiss.IndexFlatIP(vectors.shape[1])
                index.add(vectors)
                with atomic_file(destination) as temporary:
                    faiss.write_index(index, str(temporary))
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
