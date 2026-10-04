# Evaluation runner

Copy the new `scripts/evaluate.py` and `scripts/evaluation.py` into `track_2a/scripts/`.
Also merge the updated `scripts/check_claim.py` (optional embedding-model cache) and
`scripts/apertus_nli.py` (preserves API token usage on rejected output).
The existing `booklet_manifest.py` and `sparse_index.py` are required.
No new dependency is needed to evaluate an existing JSONL export.

## Export the dataset once

The runner accepts the schema shown in the dataset viewer:
`claim`, `claim_language`, `entailment_label`, `booklet_url`, and optional
`reference_string` / `reference_language`. The confirmed labels are 0=ENTAIL,
1=NEUTRAL, 2=CONTRADICT. Canonical string labels also work. Custom schemas can use
`--claim-field`, `--label-field` and a JSON `--label-map` file.

If you already have a JSONL export, use that directly. Otherwise, run from `track_2a`:

```bash
uv add datasets
uv run python - <<'PY'
import json
from pathlib import Path
from datasets import load_dataset

dataset = load_dataset("OSTswiss/MNLIoverSwissVotingBooklets", "default", split="train")
path = Path("data/evaluation/train.jsonl")
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("x", encoding="utf-8") as output:
    for index, row in enumerate(dataset):
        output.write(json.dumps({**row, "config": "default", "split": "train",
                                 "row_index": index}, ensure_ascii=False) + "\n")
print(f"Saved {len(dataset)} examples to {path}")
PY
```

Loading uses the documented Hugging Face [`load_dataset` API](https://huggingface.co/docs/datasets/loading).
You can pass `revision="COMMIT_SHA"` to load a known dataset revision. Preserve the
export and `uv.lock`; each evaluation records the export's content hash. The export
command refuses to overwrite an existing file. If HF authentication is needed,
configure it locally; no credentials belong in exports or run configuration.

Keep original row indices if you export a subset. The runner never interprets the
line number of a filtered file as a dataset row index. It joins `booklet_url` through
`data/booklets/manifest.json` to the PDF content hash. Alternatively, a row may supply
`document_id`, or its original `row_index` with config/split. If multiple identifiers
are supplied they must all agree. No document path is inferred from a filename hash.

## First checkpoint: retrieval only

```bash
uv run python scripts/evaluate.py \
  --input data/evaluation/train.jsonl --split train \
  --limit 10 --seed 42 --retrieval-only \
  --output-dir data/evaluation/runs/retrieval-smoke
```

This makes no CSCS request. It retrieves up to five chunks per selected claim and
checks metadata integration. All relevant booklets must already be indexed, and
their BM25 caches must be current. The run records errors instead of skipping cases.

## Classification evaluation

```bash
uv run --env-file .env python scripts/evaluate.py \
  --input data/evaluation/train.jsonl --split train \
  --limit 10 --seed 42 \
  --output-dir data/evaluation/runs/hybrid-smoke
```

There is at most one Apertus call per selected claim, with no automated API retries.
No-evidence cases retain the existing NEUTRAL/no_evidence behavior without a call.
The encoder is reused across claims with the same model/revision/device. Qdrant and
BM25 validation still happen per claim. Indexes remain unchanged.

Use `--retrieval dense` with the same input, seed and limit and a new output directory
to compare with the dense-only baseline. Use a larger `--limit` after checking the
small run. `--split train` records the actual source split; it does not create a new
held-out evaluation partition. The runner rejects mixed source split/config rows.

## Outputs and interpretation

### Prompt comparison after the first smoke run

The default prompt is now `voting-nli-v2`. It requests a compact explanation,
only necessary short contiguous citations with their correct IDs, and checks for
negation, attribution and proposal scope. Retrieval and the 768-token output budget
are unchanged for this experiment. Citation validation is not relaxed. Wrong labels
are not automatically corrected from the evaluation dataset.

Run the same export/seed/limit in a new output directory:

```bash
APERTUS_PROMPT_VERSION=voting-nli-v2 uv run --env-file .env python scripts/evaluate.py \
  --input data/evaluation/train.jsonl --split train --limit 10 --seed 42 \
  --output-dir data/evaluation/runs/hybrid-prompt-v2
```

The original prompt remains selectable with `APERTUS_PROMPT_VERSION=voting-nli-v1`.
Compare error count, coverage, accuracy, completion tokens and inference time. A
successful quote check does not guarantee a correct label. These ten examples have
now informed development, so later quality claims need a separate untouched sample.

Failure diagnostics now retain `finish_reason` and partial `raw_model_output` for
non-stop completions. An invalid citation includes `matching_evidence_ids` to show
whether its quote belongs to another supplied passage; the code does not silently
change the citation ID or accept it. No partial JSON repair or automatic retry is used.

## Saved run files

### Experimental passage selection instead of generated quotes

`APERTUS_PROMPT_VERSION=voting-nli-v3-ids` changes the citation output contract:
the model selects supplied passage IDs, and Python attaches the original full passage
text locally. It retains the v2 task/label instructions, model, temperature, retrieval
and 768-token output budget. This is an optional experiment; the default remains v2.
The v1 and v2 request bodies match the previously saved request hashes.

```bash
APERTUS_PROMPT_VERSION=voting-nli-v3-ids uv run --env-file .env python scripts/evaluate.py \
  --input data/evaluation/train.jsonl --split train --limit 10 --seed 42 \
  --output-dir data/evaluation/runs/hybrid-passage-ids
```

Update `scripts/apertus_nli.py` to use this mode. Other runner files from the previous
step are compatible. Responses contain the same label and explanation fields, with
`citations: [{"evidence_id": "E2"}]`. ENTAIL/CONTRADICT still require evidence;
NEUTRAL permits an empty list. Unknown IDs, duplicate selections, extra citation
fields, malformed responses and incomplete completions are rejected.

Resolved citations expose `citation_scope: whole_passage`, `quote_origin: local_source`,
`match_type: selected_passage` and `model_quote: null`. The `quote` field contains the
exact full source text for compatibility, not a span extracted by the model. Source
URLs, PDF page numbers and chunk IDs continue to come from the local evidence.
`requires_review: false` means no source word-spacing repair occurred; it does not
certify evidence relevance, entailment, or correctness of the explanation.

Compare error types, completion length and classification accuracy separately.
Selecting an existing but irrelevant passage is still possible. Negation errors and
insufficient retrieval can persist after all citation-copying errors disappear.
Do not retroactively score failed quote-mode responses as successful ID-mode results.
See `experiments/prompt-v2-comparison.md` for the motivating development run.

Each new run folder contains:

- `config.json`: input and manifest hashes, label mapping, selected IDs, seed,
  retrieval settings, API configuration (no key), prompt version, package/Python
  versions, source code hashes and booklet metadata.
- `predictions.jsonl`: one flushed record per completed claim, containing gold label,
  predicted label/explanation/citations, retrieved passages, retrieval diagnostics,
  timings, API usage or explicit failure details. Gold labels, `reference_string`,
  gold chunk IDs and dataset references never enter the query or NLI prompt.
- `summary.json`: overall and per-claim-language metrics, coverage, errors and resource
  totals. Claim language, rather than reference language, defines language groups.

Accuracy divides correct predictions by all processed examples; failed calls or
invalid outputs count as incorrect. `accuracy_on_valid_predictions` reports the
successful-prediction subset separately; `prediction_coverage` shows its size.
Macro-F1 averages all three classes, with zero for an absent class. The confusion
matrix has gold labels as rows and predictions (including `ERROR`) as columns.
Citation-review flags are counted but do not remove a prediction from metrics.

`prompt_tokens` is the API's input/context token usage, including the system prompt,
claim and supplied evidence. Completion and total tokens are also reported. Missing
API usage stays null; sums cover only reporting calls and include a missing-call
count. Failed API requests can consume unreported tokens. Inference time is client
wall-clock time including network and validation, not server-only compute time.
Retrieval is timed separately, including initial model loading.

Recall@5 is a macro average of `|gold chunk IDs intersect top 5| / |gold chunk IDs|`.
To enable it, add a nonempty `relevant_chunk_ids` list to annotated input rows, using
IDs from the exact indexed chunk version. Missing annotations use null; an empty
list is rejected. Rows without annotations do not enter recall's denominator.
Retrieval failures on annotated examples count as zero; classification failures
retain independently computed retrieval scores. Duplicate retrieved IDs count once.

The native dataset's `reference_string` is preserved for error analysis, but is not
automatically mapped to chunks. PDF extraction artifacts, split passages and multiple
supporting passages make exact text alignment unreliable. Without verified chunk
annotations, Recall@5 is null with an explanation, not a fabricated score. Mapping
those references to indexed chunks is a separate next step.

Sampled `train` results are development diagnostics. Do not present them as held-out
performance when these examples informed prompt or retrieval tuning. Preserve a
separate evaluation set; group related claims/translations by vote when creating
development and held-out subsets. Sampling by seed alone does not prevent leakage.

## Failure behavior and verification

The output directory must be new. Missing manifests, unknown labels, conflicting
booklet IDs and malformed inputs fail before inference. Per-example retrieval/API
failures are recorded and processing continues; exit status is 1 if any example
failed. Interrupting with Ctrl-C produces a partial summary, marked `complete: false`,
and exit status 130. Completed rows remain saved. This first runner does not resume
an interrupted batch; choose a new directory for another run.

```bash
uv run python -m unittest discover -s tests -v
```

Offline tests check the supplied schema/label mapping, multilingual grouping,
hand-calculated metrics, failure denominators, token usage completeness, model reuse,
and batch output behavior. No live dataset download or CSCS evaluation was run here.
