"""Extract local voting PDFs with Docling; run from the track_2a root."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter

LOG = logging.getLogger(__name__)


def discover_pdfs(source: Path) -> list[Path]:
    """Accept one PDF or recursively discover PDFs in a directory."""
    if source.is_file() and source.suffix.lower() == ".pdf":
        return [source]
    if source.is_dir():
        return sorted(
            path for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() == ".pdf"
        )
    raise ValueError(f"Expected a PDF file or directory: {source}")


def build_converter(ocr: bool) -> DocumentConverter:
    """Create one converter, reused across the entire batch."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(do_ocr=ocr, do_table_structure=True)
    if ocr:
        options.ocr_options = EasyOcrOptions(lang=["de", "fr", "it", "en"])
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)},
    )


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_pdf(
    converter: DocumentConverter,
    pdf: Path,
    destination: Path,
    *,
    ocr: bool,
) -> int:
    """Persist native structure plus a page preview; reject partial conversions."""
    from docling.datamodel.base_models import ConversionStatus

    result = converter.convert(pdf, raises_on_error=False)
    if result.status != ConversionStatus.SUCCESS:
        raise RuntimeError(f"Conversion status {result.status}: {result.errors}")

    document = result.document
    pages = [
        {"page_number": number, "markdown": document.export_to_markdown(page_no=number)}
        for number in sorted(document.pages)
    ]
    if not pages:
        raise RuntimeError("Docling returned no pages")
    for page in pages:
        if not page["markdown"].strip():
            LOG.warning("%s: PDF page %s has no text; inspect it", pdf.name, page["page_number"])

    with pdf.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    preview = {
        "schema_version": 1,
        "source_pdf": str(pdf.resolve()),
        "source_sha256": source_hash,
        "language": None,  # Attach verified language from the manifest in a later step.
        "python_version": platform.python_version(),
        "docling_version": version("docling"),
        "docling_core_version": version("docling-core"),
        "options": {"ocr": ocr, "ocr_languages": ["de", "fr", "it", "en"] if ocr else [],
                    "table_structure": True},
        "page_count": len(pages),
        "pages": pages,
    }
    # Export everything before writing. Native JSON is the input for later chunking.
    native = document.export_to_dict()
    markdown = document.export_to_markdown()
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "document.json", native)
    (destination / "document.md").write_text(markdown + "\n", encoding="utf-8")
    write_json(destination / "pages.json", preview)
    return len(pages)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/booklets/pdfs"),
                        help="A PDF file or a directory (searched recursively)")
    parser.add_argument("--output", type=Path, default=Path("data/booklets/parsed"))
    parser.add_argument("--limit", type=int, help="Process only the first N PDFs in sorted order")
    parser.add_argument("--ocr", action="store_true", help="Enable multilingual EasyOCR for scans")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    try:
        pdfs = discover_pdfs(args.input)
    except ValueError as error:
        parser.error(str(error))
    if not pdfs:
        parser.error(f"No PDFs found under {args.input}")
    pdfs = pdfs[:args.limit]
    converter = build_converter(args.ocr)

    root = args.input if args.input.is_dir() else args.input.parent
    failed = 0
    for index, pdf in enumerate(pdfs, start=1):
        # Keep relative directories AND the .pdf suffix to avoid stem collisions.
        destination = args.output / pdf.relative_to(root)
        LOG.info("[%s/%s] Parsing %s", index, len(pdfs), pdf)
        try:
            page_count = parse_pdf(converter, pdf, destination, ocr=args.ocr)
        except Exception:
            LOG.exception("Failed to parse %s", pdf)
            failed += 1
            continue
        LOG.info("Saved %s pages to %s", page_count, destination)
    LOG.info("Finished: %s succeeded, %s failed", len(pdfs) - failed, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
