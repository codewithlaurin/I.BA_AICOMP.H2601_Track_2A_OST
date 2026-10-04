# Parse the voting booklets

This step implements PDF → Docling extraction, before our Chunk model and retrieval.
Copy `scripts/parse_booklets.py` and `requirements-docling.txt` into your existing
`track_2a` folder, preserving the `scripts/` directory. Run from that folder.
The downloader and its manifest do not need to change.

## First run

Use Python 3.11 or newer in your uv project. If there is no `pyproject.toml` yet,
initialize the project with `uv init` first:

```sh
uv add -r requirements-docling.txt
uv run python scripts/parse_booklets.py --limit 1
```

uv records the dependencies in `pyproject.toml` and `uv.lock` and manages the
environment. Manual activation is unnecessary. The script lets import errors
propagate with their original traceback rather than printing installation advice.

The first conversion downloads model weights and can take several minutes.
Processing runs locally. Network access is needed for installation and initial
model downloads. OCR is initially disabled: this assumes PDFs with embedded,
selectable text. Layout and table recognition still use models without OCR.

After inspecting the first result, process the collection:

```sh
uv run python scripts/parse_booklets.py
```

Select an individual booklet, or enable OCR for scanned content:

```sh
uv run python scripts/parse_booklets.py --input "data/booklets/pdfs/example.pdf"
uv run python scripts/parse_booklets.py --limit 1 --ocr --output data/booklets/parsed-ocr
```

Replace `example.pdf` with an actual filename. `--input` also accepts another
directory. All relative paths are relative to your terminal's working directory.
`--limit 1` processes one whole PDF, not one page.

## Outputs and design

For `data/booklets/pdfs/de/example.pdf`, a directory named
`data/booklets/parsed/de/example.pdf/` contains:

- `document.json`: native Docling output, retaining extracted structure, tables,
  hierarchy and item provenance (`prov`, including page numbers and bounding boxes).
  This is the authoritative parsed input for the next chunking step.
- `document.md`: human-readable document text and tables for inspection.
- `pages.json`: a preview containing Markdown per physical PDF page, source path,
  SHA-256, Python/Docling versions and explicit parser settings.

The script creates one converter for the batch and handles each PDF independently.
It rejects partial conversions, logs failed files and exits with a nonzero status
if any conversion fails. Empty page exports trigger warnings; an illustration-only
page may legitimately be empty. A successful conversion is not proof of accuracy.

German, French and Italian text is preserved as UTF-8 without translation.
Optional EasyOCR uses `de`, `fr`, `it` and `en`. These are recognition languages,
not detected document language. `language` remains null until we inspect and join
your manifest; no vote date, source URL or language is guessed from filenames.

Page numbers are physical PDF positions, usually starting at 1; they need not match
printed page labels. The Markdown previews are not final retrieval chunks. Native
JSON preserves details that Markdown omits, including page furniture. Diagrams are
not semantically interpreted by this baseline.

Rerunning explicitly reparses selected PDFs and replaces their generated output
files. It does not edit input PDFs or the manifest, and has no resume/cache logic
yet. Use a different `--output` directory when comparing OCR settings. Downstream
chunking will load saved JSON, so claims will not require PDF conversion again.
If writing fails, that PDF's output directory may be incomplete; rerun it before use.

## Checkpoint

Expect logs ending with `Finished: 1 succeeded, 0 failed` for the trial. Compare
`document.md` and a few entries in `pages.json` with the original PDF: check a
heading, a two-column page, accented text, and a table if present. Confirm that
page references point to the right physical pages. Later repeat for DE, FR and IT.

Common problems:

- `No PDFs found` / missing input: run from `track_2a` or provide `--input`.
- Missing Docling / OCR package: run `uv add -r requirements-docling.txt` in the
  project root and invoke the script through `uv run`.
- Model download errors: check network access, proxy/certificate setup and cache
  permissions. A failed first download may be retried on the next run.
- Empty or missing text: inspect the original and compare an OCR run; scanned
  regions can be missed with OCR off. OCR does not guarantee perfect extraction.
- Slow conversion or memory pressure: start with one booklet; the script processes
  files sequentially, but large pages and tables can still be expensive.
- Partial/failed conversion: inspect the logged exception; unsuccessful results
  are not exported as new successful documents. Older outputs can still exist.

Stop after validating one booklet. Next define the Chunk model and transform the
saved structure into page-oriented evidence chunks with verified source metadata.

## References

- [Docling converter API](https://docling-project.github.io/docling/reference/document_converter/)
- [Docling document/export API](https://docling-project.github.io/docling/reference/docling_document/)
- [Docling package](https://pypi.org/project/docling/)

The provided release is pinned. Commit `pyproject.toml` and `uv.lock` to record
resolved dependency versions for reproducible experiments. Model artifacts are
not pinned by this lockfile. Actual PDF extraction must be validated in your environment.
