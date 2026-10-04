"""Build and reuse a small, persistent BM25 index over existing Qdrant chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BM25Config:
    k1: float = 1.2
    b: float = 0.75
    tokenizer: str = "unicode-nfkc-casefold-words-v1"
    idf: str = "log1p-robertson-v1"

    def __post_init__(self) -> None:
        if not math.isfinite(self.k1) or self.k1 <= 0 or not 0 <= self.b <= 1:
            raise ValueError("BM25 requires finite k1 > 0 and 0 <= b <= 1")
        if self.tokenizer != "unicode-nfkc-casefold-words-v1" or self.idf != "log1p-robertson-v1":
            raise ValueError("Unsupported tokenizer or IDF version; rebuild the BM25 index")


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)


def corpus_fingerprint(chunks: list[dict]) -> str:
    canonical = json.dumps(sorted(chunks, key=lambda item: item["chunk_id"]),
                           sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def index_path(root: Path, collection: str, document_id: str) -> Path:
    # Hash both names, avoiding path traversal and collection-name collisions.
    key = hashlib.sha256(json.dumps([collection, document_id]).encode()).hexdigest()
    return root / f"{key}.json"


def read_qdrant_chunks(client: Any, collection: str, document_id: str | None = None) -> list[dict]:
    from qdrant_client import models

    scope = None if document_id is None else models.Filter(must=[models.FieldCondition(
        key="document_id", match=models.MatchValue(value=document_id))])
    chunks, offset = [], None
    while True:
        records, offset = client.scroll(collection_name=collection, scroll_filter=scope,
                                        offset=offset, limit=256, with_payload=True, with_vectors=False)
        for record in records:
            payload = record.payload or {}
            if str(record.id) != payload.get("chunk_id"):
                raise ValueError("Qdrant point ID differs from its chunk_id")
            if document_id is not None and payload.get("document_id") != document_id:
                raise ValueError("Qdrant returned a chunk outside the selected document")
            if not isinstance(payload.get("text"), str) or not payload["text"].strip():
                raise ValueError(f"Chunk {record.id} has no usable text")
            chunks.append(payload)
        if offset is None:
            break
    return sorted(chunks, key=lambda item: item["chunk_id"])


class BM25Index:
    """JSON stores postings, lengths and corpus provenance; loading never retokenizes."""

    def __init__(self, data: dict):
        if data.get("schema_version") != 1:
            raise ValueError("Unsupported BM25 index schema; rebuild it")
        self.data = data
        self.config = BM25Config(**data["config"])

    @classmethod
    def build(cls, chunks: list[dict], collection: str, config: BM25Config) -> BM25Index:
        if not chunks:
            raise ValueError("Cannot index an empty document")
        chunks = sorted(chunks, key=lambda item: item["chunk_id"])
        document_ids = {item["document_id"] for item in chunks}
        if len(document_ids) != 1 or len({item["chunk_id"] for item in chunks}) != len(chunks):
            raise ValueError("Expected unique chunks belonging to one document")
        if len({item["parsed_sha256"] for item in chunks}) != 1:
            raise ValueError("Document contains mixed parsed snapshots; finish vector indexing first")
        postings: dict[str, list[list[int]]] = defaultdict(list)
        documents = []
        for position, chunk in enumerate(chunks):
            frequencies = Counter(tokenize(chunk["text"]))
            documents.append({"chunk_id": chunk["chunk_id"], "page_number": chunk["page_number"],
                              "part": chunk["part"], "length": sum(frequencies.values())})
            for term, frequency in frequencies.items():
                postings[term].append([position, frequency])
        total_tokens = sum(item["length"] for item in documents)
        if not total_tokens:
            raise ValueError("Document contains no searchable tokens")
        return cls({"schema_version": 1, "collection": collection,
                    "document_id": next(iter(document_ids)), "config": asdict(config),
                    "corpus_sha256": corpus_fingerprint(chunks), "documents": documents,
                    "average_length": total_tokens / len(documents), "postings": dict(postings)})

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        if not path.is_file():
            raise ValueError("BM25 index missing; run scripts/sparse_index.py first")
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.data, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(path)

    def validate(self, chunks: list[dict], collection: str, document_id: str) -> None:
        if (self.data["collection"] != collection or self.data["document_id"] != document_id
                or self.data["corpus_sha256"] != corpus_fingerprint(chunks)):
            raise ValueError("BM25 index is stale or belongs to another corpus; run scripts/sparse_index.py again")

    def search(self, query: str, limit: int = 50) -> list[dict]:
        if limit < 1:
            raise ValueError("BM25 candidate limit must be positive")
        documents, postings = self.data["documents"], self.data["postings"]
        scores: dict[int, float] = defaultdict(float)
        for term in sorted(set(tokenize(query))):
            matches = postings.get(term, [])
            if not matches:
                continue
            idf = math.log1p((len(documents) - len(matches) + 0.5) / (len(matches) + 0.5))
            for position, frequency in matches:
                length = documents[position]["length"]
                normalization = self.config.k1 * (
                    1 - self.config.b + self.config.b * length / self.data["average_length"])
                scores[position] += idf * frequency * (self.config.k1 + 1) / (frequency + normalization)
        ranked = sorted(scores, key=lambda pos: (-scores[pos], documents[pos]["chunk_id"]))[:limit]
        return [{"chunk_id": documents[pos]["chunk_id"], "page_number": documents[pos]["page_number"],
                 "part": documents[pos]["part"], "bm25_rank": rank, "bm25_score": scores[pos]}
                for rank, pos in enumerate(ranked, start=1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/qdrant"))
    parser.add_argument("--collection", default="voting_booklets_e5_v1")
    parser.add_argument("--output", type=Path, default=Path("data/bm25"))
    parser.add_argument("--document-id", help="Build one booklet only; default is all indexed booklets")
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()
    config = BM25Config(k1=args.k1, b=args.b)
    if not args.db.is_dir():
        parser.error("Qdrant database does not exist; finish vector indexing first")
    from qdrant_client import QdrantClient

    client = QdrantClient(path=str(args.db))
    try:
        if not client.collection_exists(args.collection):
            raise ValueError("Collection does not exist")
        groups: dict[str, list[dict]] = defaultdict(list)
        for chunk in read_qdrant_chunks(client, args.collection, args.document_id):
            groups[chunk["document_id"]].append(chunk)
        if not groups:
            raise ValueError("No indexed chunks found; check the collection and document ID")
        for document_id, chunks in sorted(groups.items()):
            path = index_path(args.output, args.collection, document_id)
            if path.is_file():
                existing = BM25Index.load(path)
                if existing.config == config and existing.data["corpus_sha256"] == corpus_fingerprint(chunks):
                    print(f"Unchanged: {document_id} ({len(chunks)} chunks)")
                    continue
            BM25Index.build(chunks, args.collection, config).save(path)
            print(f"Indexed: {document_id} ({len(chunks)} chunks) → {path}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
