"""Cached BM25 inverted index over full booklet pages, without model inference."""

from collections import Counter
from dataclasses import dataclass
import json
import math
import re
import unicodedata

from .cache import atomic_file, cache_dir, cache_lock, content_key

SPARSE_VERSION = 1  # Bump when tokenization or stored index format changes.
BM25_K1 = 1.2
BM25_B = 0.75
_TOKENS = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+", re.UNICODE)
_INDEXES: dict[str, "BM25Index"] = {}


def tokenize(text: str) -> list[str]:
    """Unicode words and numbers for de/fr/it; normalize case and numeric separators."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"(?<=\d)['’](?=\d)", "", text)
    return [token.replace(",", ".") for token in _TOKENS.findall(text)]


@dataclass
class BM25Index:
    lengths: list[int]
    postings: dict[str, list[list[int]]]

    @classmethod
    def build(cls, pages: tuple) -> "BM25Index":
        lengths, postings = [], {}
        for index, page in enumerate(pages):
            tokens = tokenize(page["text"])
            lengths.append(len(tokens))
            for term, frequency in Counter(tokens).items():
                postings.setdefault(term, []).append([index, frequency])
        return cls(lengths, postings)

    def search(self, query: str, limit: int) -> list[tuple[int, float]]:
        """Return matching pages only, with positive BM25 scores, best first.

        Uses Lucene's positive IDF: log(1 + (N - df + 0.5) / (df + 0.5)).
        No lexical match contributes no ranking to RRF, preserving dense search
        for translated claims and queries with no shared words.
        """
        count = len(self.lengths)
        average = sum(self.lengths) / count if count else 0
        if not average or limit <= 0:
            return []
        scores: dict[int, float] = {}
        for term, query_frequency in Counter(tokenize(query)).items():
            hits = self.postings.get(term, [])
            if not hits:
                continue
            idf = math.log1p((count - len(hits) + 0.5) / (len(hits) + 0.5))
            for index, frequency in hits:
                norm = BM25_K1 * (1 - BM25_B + BM25_B * self.lengths[index] / average)
                score = query_frequency * idf * frequency * (BM25_K1 + 1) / (frequency + norm)
                scores[index] = scores.get(index, 0.0) + score
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]


def _save_index(destination, index: BM25Index) -> None:
    data = {"lengths": index.lengths, "postings": index.postings}
    with atomic_file(destination) as temporary:
        temporary.write_text(json.dumps({"data": data, "checksum": content_key(data)},
                                        ensure_ascii=False), encoding="utf-8")


def booklet_bm25(pages: tuple) -> BM25Index:
    """Reuse content-addressed term statistics across claims and process restarts."""
    digest = content_key({"version": SPARSE_VERSION, "pages": pages})
    destination = cache_dir() / "sparse" / (digest + ".bm25.json")
    key = str(destination)
    if key in _INDEXES:
        index = _INDEXES[key]
        if not destination.is_file():
            with cache_lock(destination):
                if not destination.is_file():
                    _save_index(destination, index)
        return index

    with cache_lock(destination):
        try:
            stored = json.loads(destination.read_text(encoding="utf-8"))
            data = stored["data"]
            if (stored["checksum"] != content_key(data) or len(data["lengths"]) != len(pages)
                    or any(type(length) is not int or length < 0 for length in data["lengths"])
                    or not isinstance(data["postings"], dict)):
                raise ValueError("invalid cached BM25 index")
            index = BM25Index(data["lengths"], data["postings"])
        except (OSError, ValueError, KeyError, TypeError):
            index = BM25Index.build(pages)
            _save_index(destination, index)
    if len(_INDEXES) >= 16:
        _INDEXES.pop(next(iter(_INDEXES)))
    _INDEXES[key] = index
    return index
