"""Index parsed booklets in persistent local Qdrant, or search the saved vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from booklet_chunks import Chunk, load_chunks

if TYPE_CHECKING:
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexConfig:
    model: str = "intfloat/multilingual-e5-small"
    revision: str = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    dimensions: int = 384
    max_tokens: int = 480  # Includes E5 prefix and special tokens; model limit is 512.
    passage_prefix: str = "passage: "
    query_prefix: str = "query: "
    chunker: str = "page-recursive-v1"

    @property
    def vector_name(self) -> str:
        fingerprint = hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()
        return f"e5_{fingerprint[:16]}"


CONFIG = IndexConfig()


def load_encoder(device: str) -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(CONFIG.model, revision=CONFIG.revision, device=device)
    if encoder.get_sentence_embedding_dimension() != CONFIG.dimensions:
        raise ValueError("Embedding dimension does not match the index configuration")
    if encoder.max_seq_length < CONFIG.max_tokens:
        raise ValueError("Chunk token budget exceeds the model's input limit")
    return encoder


def token_count(encoder: SentenceTransformer, text: str) -> int:
    return len(encoder.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])


def ensure_collection(client: QdrantClient, collection: str, *, create: bool) -> None:
    from qdrant_client import models

    if not client.collection_exists(collection):
        if not create:
            raise ValueError(f"Collection {collection!r} does not exist; run index first")
        client.create_collection(collection_name=collection, vectors_config={
            CONFIG.vector_name: models.VectorParams(size=CONFIG.dimensions, distance=models.Distance.COSINE)
        })
    vectors = client.get_collection(collection).config.params.vectors
    if not isinstance(vectors, dict) or set(vectors) != {CONFIG.vector_name}:
        raise ValueError("Collection uses another model/chunker configuration; choose a new --collection")
    params = vectors[CONFIG.vector_name]
    if params.size != CONFIG.dimensions or params.distance != models.Distance.COSINE:
        raise ValueError("Collection vector size or distance does not match configuration")


def index_document(
    client: QdrantClient, collection: str, encoder: SentenceTransformer,
    chunks: list[Chunk], batch_size: int,
) -> bool:
    """Resume unchanged documents; upsert batches, then remove obsolete page parts."""
    from qdrant_client import models

    first = chunks[0]
    document_condition = models.FieldCondition(
        key="document_id", match=models.MatchValue(value=first.document_id))
    document_filter = models.Filter(must=[document_condition])
    current_filter = models.Filter(must=[document_condition, models.FieldCondition(
        key="parsed_sha256", match=models.MatchValue(value=first.parsed_sha256))])
    total = client.count(collection, count_filter=document_filter, exact=True).count
    current = client.count(collection, count_filter=current_filter, exact=True).count
    if total == current == len(chunks):
        LOG.info("Unchanged: %s (%s chunks)", first.source_pdf, total)
        return False

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        texts = [CONFIG.passage_prefix + chunk.text for chunk in batch]
        vectors = encoder.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 show_progress_bar=False)
        points = [models.PointStruct(
            id=chunk.chunk_id, vector={CONFIG.vector_name: vector.tolist()},
            payload={**chunk.payload(), "index_config": asdict(CONFIG)},
        ) for chunk, vector in zip(batch, vectors, strict=True)]
        client.upsert(collection_name=collection, points=points, wait=True)

    # Delete only obsolete parts of this same PDF, after all new vectors are written.
    stale = models.Filter(must=[document_condition], must_not=[
        models.HasIdCondition(has_id=[chunk.chunk_id for chunk in chunks])])
    client.delete(collection_name=collection,
                  points_selector=models.FilterSelector(filter=stale), wait=True)
    LOG.info("Indexed: %s (%s chunks)", first.source_pdf, len(chunks))
    return True


def search(
    client: QdrantClient, collection: str, encoder: SentenceTransformer,
    query: str, top_k: int, document_id: str | None,
) -> None:
    from qdrant_client import models

    text = CONFIG.query_prefix + query.strip()
    if not query.strip() or token_count(encoder, text) > encoder.max_seq_length:
        raise ValueError("Query must be nonempty and fit the model's token limit")
    query_filter = None
    if document_id:
        query_filter = models.Filter(must=[models.FieldCondition(
            key="document_id", match=models.MatchValue(value=document_id))])
    vector = encoder.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
    hits = client.query_points(collection_name=collection, query=vector.tolist(),
                               using=CONFIG.vector_name, query_filter=query_filter,
                               limit=top_k, with_payload=True).points
    for rank, hit in enumerate(hits, start=1):
        payload = hit.payload or {}
        print(json.dumps({"rank": rank, "score": hit.score, **payload}, ensure_ascii=False))
    if not hits:
        LOG.info("No matching chunks")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/qdrant"))
    parser.add_argument("--collection", default="voting_booklets_e5_v1")
    parser.add_argument("--device", default="cpu", help="cpu (default), mps, or cuda")
    commands = parser.add_subparsers(dest="command", required=True)
    indexing = commands.add_parser("index", help="Embed Docling page exports")
    indexing.add_argument("--input", type=Path, default=Path("data/booklets/parsed"))
    indexing.add_argument("--limit", type=int, help="Maximum number of booklets")
    indexing.add_argument("--batch-size", type=int, default=16)
    searching = commands.add_parser("search", help="Dense retrieval smoke test")
    searching.add_argument("query")
    searching.add_argument("--top-k", type=int, default=5)
    searching.add_argument("--document-id", help="Restrict to one source_sha256 from pages.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    paths = []
    if args.command == "index":
        if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
            parser.error("--batch-size and --limit must be positive")
        paths = ([args.input] if args.input.is_file()
                 else sorted(args.input.rglob("pages.json")))
        paths = paths[:args.limit]
        if not paths:
            parser.error(f"No pages.json files found in {args.input}; run parsing first")
    elif args.top_k < 1 or not args.query.strip():
        parser.error("Use a nonempty query and positive --top-k")
    elif not args.db.is_dir():
        parser.error("Local database does not exist; run index first")

    from qdrant_client import QdrantClient

    client = QdrantClient(path=str(args.db))
    try:
        ensure_collection(client, args.collection, create=args.command == "index")
        encoder = load_encoder(args.device)
        if args.command == "search":
            search(client, args.collection, encoder, args.query, args.top_k, args.document_id)
        else:
            indexed = skipped = 0
            for path in paths:
                LOG.info("Reading %s", path)
                chunks = load_chunks(path, lambda text: token_count(
                    encoder, CONFIG.passage_prefix + text) <= CONFIG.max_tokens)
                if index_document(client, args.collection, encoder, chunks, args.batch_size):
                    indexed += 1
                else:
                    skipped += 1
            count = client.count(args.collection, exact=True).count
            LOG.info("Finished: %s indexed, %s unchanged; %s total chunks", indexed, skipped, count)
    finally:
        client.close()


if __name__ == "__main__":
    main()
