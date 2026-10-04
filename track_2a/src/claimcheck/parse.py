"""Stufe 1: PDF → DoclingDocument (JSON) + Markdown zur Kontrolle.

Aufruf als Modul:   from claimcheck.parse import parse_pdf
Aufruf als Script:  python -m claimcheck.parse data/pdf/buechli.pdf [weitere.pdf ...]
"""
import logging
import sys
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, granite_picture_description
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument

from . import config

log = logging.getLogger(__name__)


def build_converter(
    describe_pictures: bool = config.DESCRIBE_PICTURES,
    device: AcceleratorDevice = AcceleratorDevice.AUTO,
) -> DocumentConverter:
    """Docling-Converter mit unseren Einstellungen. Einmal bauen, mehrfach verwenden."""
    opts = PdfPipelineOptions()
    opts.accelerator_options = AcceleratorOptions(device=device, num_threads=16)
    opts.do_ocr = False                      # Text-PDFs, keine OCR nötig
    opts.images_scale = config.IMAGES_SCALE
    opts.generate_picture_images = True

    if describe_pictures:
        opts.do_picture_description = True
        opts.picture_description_options = granite_picture_description
        opts.picture_description_options.prompt = (
            "Describe the image in three sentences. Be concise and accurate. "
            "Name categories and numbers if the image is a chart."
        )
        opts.picture_description_options.picture_area_threshold = config.PICTURE_AREA_THRESHOLD

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def parse_pdf(
    pdf: Path,
    out_dir: Path = config.PARSED_DIR,
    converter: DocumentConverter | None = None,
) -> Path:
    """Parst ein PDF und speichert <name>.docling.json und <name>.md in out_dir.

    Gibt den Pfad zur JSON-Datei zurück. Existiert sie schon, wird nicht neu geparst.
    """
    pdf = Path(pdf)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{pdf.stem}.docling.json"
    out_md = out_dir / f"{pdf.stem}.md"

    if out_json.exists():
        log.info("%s bereits geparst, überspringe", pdf.name)
        return out_json

    converter = converter or build_converter()
    doc = converter.convert(str(pdf)).document
    doc.save_as_json(out_json)
    out_md.write_text(doc.export_to_markdown(), encoding="utf-8")

    n_desc = sum(1 for p in doc.pictures if p.annotations)
    log.info(
        "%s: %d Seiten, %d Tabellen, %d Bilder (%d beschrieben) -> %s",
        pdf.name, len(doc.pages), len(doc.tables), len(doc.pictures), n_desc, out_json.name,
    )
    return out_json


def load_parsed(json_path: Path) -> DoclingDocument:
    """Lädt ein gespeichertes DoclingDocument wieder (für chunks.py)."""
    return DoclingDocument.load_from_json(Path(json_path))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if len(sys.argv) < 2:
        sys.exit("Usage: python -m claimcheck.parse <pdf> [<pdf> ...]")
    conv = build_converter()
    for arg in sys.argv[1:]:
        parse_pdf(Path(arg), converter=conv)
