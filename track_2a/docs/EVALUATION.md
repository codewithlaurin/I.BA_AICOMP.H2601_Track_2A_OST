# Judge-compatible evaluation

The interface follows the [official starter](https://gitlab.com/ifsoftware/hackapertus-starter/)
and [Solution API Guide](https://notes.rehash.ch/s/7912e1d2-5d79-4c93-add9-12ec4a6d1b1e),
checked on 2026-10-06. Commands below run from `track_2a/`.

## Predict

```bash
uv run --env-file .env python main.py --input data/cases.jsonl --output output/predictions.jsonl
```

The entrypoint accepts mixed task A (PDF) and task B (reference passage) JSONL.
Relative booklet paths resolve against the input file's directory (`/data` in the
judges' container); `--data-root` overrides this for local runs. Inputs must have
unique string IDs, a vote, claim text/language, and exactly one source. Languages
are `de`, `fr`, or `it`, in any source/claim combination.

The judge baseline sends all extracted PDF pages for A, or the supplied reference
for B, plus the vote and claim to Apertus. It uses CPU PDFium text extraction and
caches at most 16 PDFs in memory. It requires no prebuilt Qdrant database, embedding
weights, manifest, or runtime downloads. The existing Docling/BM25/dense experiments
remain available through `scripts/check_claim.py`; they are not this baseline.

Runtime environment, in precedence order:

| Setting | Variables | Default |
| --- | --- | --- |
| Endpoint | `BASE_URL`, `LLM_BASE_URL`, `CSCS_INFERENCE_BASE_URL` | `https://api.inference.cscs.ch/v1` |
| Key | `API_KEY`, `LLM_API_KEY`, `CSCS_INFERENCE_API_KEY` | Required |
| Model | `LLM_NAME`, `APERTUS_MODEL` | `swiss-ai/Apertus-v1.5-8B` |

The entrypoint rejects older Apertus models. If your existing `.env` selects
`Apertus-8B-Instruct-2509`, update it or set `LLM_NAME=swiss-ai/Apertus-v1.5-8B`.
Injected `BASE_URL` and `API_KEY` always win, including when empty (an empty value
is an error). Python does not load `.env` implicitly; use uv's explicit flag locally.

Responses echo the ID and use `0/entailment`, `1/neutral`, or `2/contradiction`.
Evidence contains short verbatim `{ "page": ..., "text": ... }` excerpts, with
1-based physical PDF pages for A and `null` for B. Entailment/contradiction require
at least one quote. Quotes are resolved back to source text using the existing
citation validator; emitted text is always an exact source substring. Whole source
passages and quotes exceeding 500 characters are rejected. This length is a local
implementation cap, not a published numeric judge requirement.

Exactly one API call is made per case, with temperature 0, a 1024-token output
budget, and no retries. `input_tokens` and `output_tokens` come from API usage;
completion tokens already include reasoning tokens. `inference_time_ms` covers
source loading, parsing, the API request, and validation for that case. A PDF cache
hit can reduce the time but cannot change the prompt or prediction inputs.

Malformed input fails before the output is opened. Successful runs exit 0. Per-case
failures write `{id, error, metrics}` (an intentionally invalid judge prediction),
continue the batch, and cause exit 1. Unknown token usage remains null. Completed
rows are flushed; Ctrl-C exits 130. Existing output files are overwritten, so use
new output paths to preserve prior runs. There is no automatic resume.

Citation mismatch errors also include a `diagnostics` object in `predictions.jsonl`:

```json
"diagnostics": {
  "evidence_id": "E1",
  "model_quote": "La pensione aumenta.",
  "source_text": "Die Rente steigt. Weitere Angaben fehlen.",
  "chunk_id": "sample:1",
  "page_number": null,
  "source_pdf": null
}
```

`model_quote` is the exact quote returned by Apertus; `source_text` is the full,
unaltered passage it was compared against, including extraction artifacts and
line breaks. PDF cases also identify the source file and its 1-based page number;
reference cases use null for these fields. These details are saved automatically
on future runs; old error-only results cannot recover the discarded quote.
The prediction still counts as an error. No additional API call is made.

## Prepare a local development sample

Use your existing dataset export; no `datasets` dependency is required:

```bash
uv run python scripts/prepare_cases.py \
  --input data/evaluation/train.jsonl \
  --output-dir data/evaluation/judge-smoke \
  --task both --limit 10 --seed 42 --download-booklets
```

Preparation writes `cases.jsonl`, `expected-labels.jsonl`, and `preparation.json` to
a new directory. It samples rows reproducibly and creates both tasks for each row.
Original `row_index` values are preserved when present; otherwise IDs use positions
in the supplied file, which must be the original export for stable dataset IDs.
The input SHA-256 and selected rows are recorded. Downloaded PDFs use URL-hash
filenames to prevent collisions. Downloads happen only during preparation.
For reference-only evaluation, use `--task B` and omit `--download-booklets`.
Title-only references are preserved. Keep gold labels outside the predictor container.

## Score predictions separately

```bash
LLM_NAME=swiss-ai/Apertus-v1.5-8B uv run --env-file .env python main.py \
  --input data/evaluation/judge-smoke/cases.jsonl \
  --output output/judge-smoke/predictions.jsonl

uv run python scripts/evaluate.py \
  --input data/evaluation/judge-smoke/cases.jsonl \
  --predictions output/judge-smoke/predictions.jsonl \
  --expected data/evaluation/judge-smoke/expected-labels.jsonl \
  --output output/judge-smoke/summary.json --verify-pdfs
```

The scorer reports macro-F1 separately for A and B, accuracy, prediction coverage,
confusion matrices, per-claim-language and source-to-claim language breakdowns, and
reported token/time totals with counts of missing values. Macro-F1 averages all
three classes, assigning 0 to undefined class F1. Missing/invalid responses count
as incorrect and as false negatives; they are never dropped from denominators.
Duplicate or unknown prediction IDs and mismatched gold IDs are rejected. Empty
task groups report null accuracy/F1. The summary records hashes of all three inputs.

Reference quotes are checked locally. `--verify-pdfs` additionally checks PDF quotes
against the stated page. This checks source fidelity, not semantic evidence quality;
the judges' evidence assessment and token-counting proxy remain authoritative.
The scorer exits 1 for missing or invalid predictions but still writes the summary.

This is a development sample, not held-out performance. Group related votes and
translations when creating a held-out split. Previously saved evaluation runs use
a different, historical schema and must not be fed directly to this scorer.

## Docker

```bash
docker build --platform linux/amd64 -t hackapertus-ost:dev .
mkdir -p output/judge-smoke
docker run --rm --platform linux/amd64 \
  -e BASE_URL -e API_KEY -e LLM_NAME \
  -v "$PWD/data/evaluation/judge-smoke/cases.jsonl:/data/cases.jsonl:ro" \
  -v "$PWD/data/evaluation/judge-smoke/booklets:/data/booklets:ro" \
  -v "$PWD/output/judge-smoke:/output" \
  hackapertus-ost:dev --input /data/cases.jsonl --output /output/predictions.jsonl
```

Export `BASE_URL` and `API_KEY` first. Omit the booklet mount for reference-only cases.
The minimal runtime installs `requirements-evaluation.txt`; it excludes the heavier
Docling/retrieval development dependencies. The build context is an allowlist and
only runtime source files are copied, excluding `.env`, datasets, and gold labels.

`make run INPUT=... DATA_DIR=... OUTPUT_DIR=...` builds and runs the same image.
Only `DATA_DIR/booklets` and the exact input file are mounted, so sibling gold-label
files remain outside the container. `BOOKLETS_DIR` overrides the booklet directory.
Defaults expect `data/cases.jsonl` and write to
`output/predictions.jsonl`. From the repository root, `make run` delegates to
`track_2a`; relative make paths are interpreted there.

## Verification and limits

```bash
uv run python -m unittest discover -s tests -v
```

Offline tests cover mixed batches, environment precedence, labels, actual token
accounting, citation validation, error denominators, order independence, input
validation, deterministic preparation, and exclusion of gold fields from prompts.

Verified on 2026-10-06: all 18 tests passed; the `linux/amd64` image built and ran
with a read-only root filesystem, read-only inputs, and writable `/tmp` and output.
A live CSCS check used one sampled dataset row (1309) in both task formats with
`swiss-ai/Apertus-v1.5-8B`. Both responses were structurally valid but predicted
neutral instead of the gold contradiction: accuracy and macro-F1 were 0 for each
task. Task A used 25,735 input / 69 output tokens in 2,896 ms; task B used 704 input /
75 output tokens in 586 ms. These two cases validate the integration, not model
quality. Local artifacts are in `output/judge-contract-smoke/`; the prepared inputs
and dataset fingerprint are in `data/evaluation/judge-contract-smoke/`. Neither
directory is committed. No prompt tuning was performed on these results.

This initial full-document baseline does not OCR image-only pages. A completely
image-only PDF fails explicitly; scanned pages inside mixed PDFs may be omitted.
Large documents fail rather than silently truncate above a 300,000-character prompt
cap; the endpoint can impose a smaller token context limit. Classification and
short-quote generation may still fail or be wrong. No automatic output repair or
fallback model hides those failures.
