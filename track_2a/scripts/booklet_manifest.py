"""Join downloader metadata to indexed booklets by PDF content SHA-256."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def load_document_metadata(path: Path, document_id: str) -> dict:
    """Read one unambiguous manifest record; never match a filename hash."""
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        raise ValueError("Manifest must contain a files list")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("Manifest must identify its dataset")
    if not re.fullmatch(r"[0-9a-f]{64}", document_id):
        raise ValueError("Document ID must be the PDF content SHA-256")
    if any(not isinstance(item, dict) for item in manifest["files"]):
        raise ValueError("Each manifest file entry must be an object")
    matches = [item for item in manifest["files"] if item.get("sha256") == document_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one manifest entry for {document_id}; found {len(matches)}")
    record = matches[0]
    url = record.get("url")
    if not isinstance(url, str) or urlsplit(url).scheme != "https" or not urlsplit(url).netloc:
        raise ValueError("Manifest PDF URL must be an absolute HTTPS URL")
    pdf_path = record.get("path")
    if not isinstance(pdf_path, str) or not pdf_path.strip():
        raise ValueError("Manifest entry has no PDF path")
    languages = record.get("languages")
    if (not isinstance(languages, list) or not languages
            or any(not isinstance(language, str) or not language.strip() for language in languages)):
        raise ValueError("Manifest languages must be a nonempty list of language codes")
    languages = sorted(set(languages))
    references = record.get("references", [])
    if not isinstance(references, list):
        raise ValueError("Manifest references must be a list")
    for reference in references:
        if (not isinstance(reference, dict)
                or any(not isinstance(reference.get(key), str) or not reference[key].strip()
                       for key in ("config", "split"))
                or type(reference.get("row_index")) is not int or reference["row_index"] < 0):
            raise ValueError("Manifest references require config, split and a nonnegative row_index")
    if type(record.get("bytes")) is not int or record["bytes"] < 1:
        raise ValueError("Manifest PDF byte count must be positive")
    if not isinstance(record.get("status"), str) or not record["status"].strip():
        raise ValueError("Manifest entry must have a download status")
    return {
        "dataset": dataset, "source_sha256": document_id, "source_url": url,
        "manifest_pdf_path": pdf_path, "languages": languages,
        "references": [{key: reference[key] for key in ("config", "split", "row_index")}
                       for reference in references],
        "download_status": record["status"], "bytes": record["bytes"],
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
    }


def attach_document_metadata(evidence: list[dict], metadata: dict) -> list[dict]:
    """Enrich returned evidence without changing Qdrant, text, scores or IDs."""
    enriched = []
    for item in evidence:
        if item["document_id"] != metadata["source_sha256"]:
            raise ValueError("Evidence and manifest identify different PDFs")
        languages = metadata["languages"]
        language = item.get("language")
        if language is not None and language not in languages:
            raise ValueError("Indexed language conflicts with manifest languages")
        if language is None and len(languages) == 1:
            language = languages[0]
        page = item["page_number"]
        if type(page) is not int or page < 1:
            raise ValueError("Evidence must identify a positive PDF page number")
        page_url = urlunsplit(urlsplit(metadata["source_url"])._replace(fragment=f"page={page}"))
        enriched.append({
            **item, "language": language, "languages": list(languages),
            "source_url": metadata["source_url"], "source_page_url": page_url,
        })
    return enriched
