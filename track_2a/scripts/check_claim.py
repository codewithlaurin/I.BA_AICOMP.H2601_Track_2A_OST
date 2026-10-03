"""Retrieve local Qdrant evidence and classify a claim with CSCS Apertus 8B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from apertus_nli import ApiConfig, api_request, build_request, classify_claim


def retrieve_evidence(
    db: Path, collection: str, document_id: str, claim: str, top_k: int, device: str,
) -> tuple[list[dict], dict]:
    from qdrant_client import QdrantClient, models
    from sentence_transformers import SentenceTransformer

    if not db.is_dir():
        raise ValueError(f"Qdrant database does not exist: {db}")
    client = QdrantClient(path=str(db))
    try:
        if not client.collection_exists(collection):
            raise ValueError(f"Collection does not exist: {collection}")
        scope = models.Filter(must=[models.FieldCondition(
            key="document_id", match=models.MatchValue(value=document_id))])
        records, _ = client.scroll(collection_name=collection, scroll_filter=scope,
                                   limit=1, with_payload=True, with_vectors=False)
        if not records:
            raise ValueError("Selected document is not indexed; check --document-id")

        # Read the stored configuration, supporting both previous indexer versions.
        config = records[0].payload["index_config"]
        fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
        vectors = client.get_collection(collection).config.params.vectors
        if not isinstance(vectors, dict) or len(vectors) != 1:
            raise ValueError("Expected one named dense vector in the collection")
        vector_name = next(iter(vectors))
        if vector_name not in {fingerprint, f"e5_{fingerprint}"}:
            raise ValueError("Stored embedding configuration does not match the collection")
        params = vectors[vector_name]
        if params.size != config["dimensions"] or params.distance != models.Distance.COSINE:
            raise ValueError("Unexpected vector dimensions or distance")

        encoder = SentenceTransformer(config["model"], revision=config["revision"], device=device)
        if encoder.get_sentence_embedding_dimension() != params.size:
            raise ValueError("Embedding model dimension differs from the stored vectors")
        query = config["query_prefix"] + claim.strip()
        tokens = encoder.tokenizer(query, add_special_tokens=True, truncation=False)["input_ids"]
        if len(tokens) > min(config["max_tokens"], encoder.max_seq_length):
            raise ValueError("Claim exceeds the embedding token budget; shorten the claim")
        vector = encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        hits = client.query_points(collection_name=collection, query=vector.tolist(),
                                   using=vector_name, query_filter=scope, limit=top_k,
                                   with_payload=True).points
        evidence = []
        for rank, hit in enumerate(hits, start=1):
            payload = hit.payload or {}
            if payload.get("index_config") != config or payload.get("document_id") != document_id:
                raise ValueError("Retrieved point has inconsistent document/model metadata")
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Retrieved point has no usable evidence text")
            evidence.append({
                "evidence_id": f"E{rank}", "chunk_id": payload["chunk_id"],
                "document_id": payload["document_id"], "source_pdf": payload["source_pdf"],
                "page_number": payload["page_number"], "part": payload["part"],
                "language": payload.get("language"), "text": text,
                "score": hit.score, "parsed_sha256": payload["parsed_sha256"],
            })
        return evidence, config
    finally:
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("claim", nargs="?")
    parser.add_argument("--document-id", help="source_sha256 of the relevant booklet")
    parser.add_argument("--db", type=Path, default=Path("data/qdrant"))
    parser.add_argument("--collection", default="voting_booklets_e5_v1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--list-models", action="store_true", help="Check models available to your CSCS key")
    parser.add_argument("--dry-run", action="store_true", help="Retrieve and show request without contacting CSCS")
    parser.add_argument("--output", type=Path, help="Also save the result as JSON")
    args = parser.parse_args()
    if not args.list_models:
        if not args.claim or not args.claim.strip() or not args.document_id:
            parser.error("Provide a claim and --document-id for the relevant voting booklet")
        if not 1 <= args.top_k <= 10:
            parser.error("--top-k must be between 1 and 10 for this PoC")

    config = ApiConfig.from_env()
    try:
        if not args.dry_run and not os.environ.get("CSCS_INFERENCE_API_KEY", "").strip():
            raise ValueError("Set CSCS_INFERENCE_API_KEY before making API requests")
        if args.list_models:
            result = api_request("models", config)
        else:
            evidence, embedding_config = retrieve_evidence(
                args.db, args.collection, args.document_id, args.claim, args.top_k, args.device)
            common = {
                "claim": args.claim, "document_id": args.document_id,
                "retrieval": {"method": "dense_only", "collection": args.collection,
                              "top_k": args.top_k, "embedding_config": embedding_config},
                "evidence": evidence,
            }
            if args.dry_run:
                result = {**common, "status": "dry_run", "request": build_request(args.claim, evidence, config)}
            else:
                result = {**common, **classify_claim(args.claim, evidence, config)}
        output = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output + "\n", encoding="utf-8")
        print(output)
        return 0
    except Exception as error:
        # Fail visibly. A technical failure must not become a NEUTRAL training label.
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
