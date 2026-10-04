# Query PoC: local Qdrant → CSCS Apertus 8B

Copy `scripts/check_claim.py`, `scripts/apertus_nli.py` and
`scripts/booklet_manifest.py` into your existing
`track_2a/scripts/`. These files work with either of the previously provided
indexer versions: the embedding model, revision and vector name are read from
Qdrant's stored configuration. The indexer does not need to change.

The default query path now uses BM25 candidates followed by dense reranking. Build
the sparse index first with `uv run python scripts/sparse_index.py`; see `BM25.md`.
Use `--retrieval dense` to retain the original dense-only PoC. In either case,
Apertus classifies using only the final retrieved passages from the selected booklet.

## API access

CSCS requires an active project with an inference resource before issuing API keys.
If the resource does not exist, the PI/deputy PI must arrange it through the route
applicable to the project's organization. Once available, a project member can
create a key in the [Inference API UI](https://ui.inference.cscs.ch/).
See the [CSCS access instructions](https://docs.cscs.ch/services/inference/api/#create-an-inference-resource).

The documented API base is `https://api.inference.cscs.ch/v1`. The default model is
`swiss-ai/Apertus-8B-Instruct-2509`. Model access depends on your key; confirm with
`--list-models`. Change `APERTUS_MODEL` if your resource uses another Apertus ID.

Copy `.env.example` to `.env`, fill in `CSCS_INFERENCE_API_KEY` locally, and exclude
`.env` from version control. No key belongs in Python source or this chat. Existing
shell environment variables also work; `--env-file` is optional if they are set.

## Run

Copy `requirements-api.txt` into `track_2a`. Install the OpenAI Python SDK using uv
(the existing Qdrant dependencies must already be installed):

```sh
uv add -r requirements-api.txt
```

Commit the updated `pyproject.toml` and `uv.lock` to record the resolved SDK version.
Then check the available models:

```sh
uv run --env-file .env python scripts/check_claim.py --list-models
```

Find the relevant booklet's `source_sha256` in its `pages.json`. Use this value as
`--document-id` so evidence from unrelated votes cannot be mixed in accidentally.
First inspect retrieval and the request without an inference call:

```sh
uv run python scripts/check_claim.py "Die E-ID ist freiwillig." \
  --document-id SOURCE_SHA256 --dry-run
```

Replace `SOURCE_SHA256` with the actual 64-character hash and choose a claim about
the indexed booklet. `--dry-run` may download embedding weights if not cached;
it does not contact the CSCS API or require its key.

Then classify:

```sh
uv run --env-file .env python scripts/check_claim.py "Die E-ID ist freiwillig." \
  --document-id SOURCE_SHA256 --output data/results/claim-001.json
```

Expected output: JSON containing a label, a brief explanation, verified citation
references, retrieved evidence with source/page information, embedding configuration,
model, prompt version, request fingerprint and usage information returned by CSCS.
The result file is replaced if you reuse its name; use distinct names for experiments.

The same command accepts German, French and Italian claims. Evidence stays in its
original language. The prompt requests the explanation in the claim's language.
This is requested behavior and must be evaluated, not assumed to be reliable.

## Manifest metadata

Queries load `data/booklets/manifest.json` by default. Override with `--manifest PATH`
or use `--no-manifest` to reproduce the previous baseline. The join uses the manifest
entry's `sha256` (PDF content hash), matching the indexed `document_id`; the hash in
the downloaded filename is not used.

The saved `document_metadata` contains the dataset, original URL, manifest PDF path,
languages, dataset references (`config`, `split`, `row_index`), download status,
byte count and a manifest content fingerprint. References belong to the booklet,
not necessarily to the current claim or cited page; they are not evaluation labels.
No voting date or title is inferred from the URL.

Evidence and locally resolved citations gain `source_url` and `source_page_url`
(`...pdf#page=N`). Page numbers refer to PDF pages; browser support for page fragments
varies. A missing passage language is filled only for a single-language manifest
record. An existing language that conflicts with the manifest raises an error.
Dataset references and download metadata are not sent to Apertus. Its evidence
still contains only the passage ID, page, language and text.

This enrichment runs after retrieval; no re-embedding or BM25 rebuild is needed,
and existing index payloads are unchanged. Missing manifests, unmatched hashes and
duplicate matching entries fail explicitly rather than guessing a booklet.

## What the code does

`check_claim.py` opens the existing local Qdrant collection, validates its stored
embedding configuration, selects BM25 candidates, embeds the claim with the same
model and prefix, and reranks those candidates within one booklet. It closes Qdrant before calling
CSCS. The classifier receives passage IDs, text, page numbers and language; local
file paths and vector scores are kept in the local result rather than sent to it.

`apertus_nli.py` defines a zero-shot NLI prompt and calls the OpenAI Python SDK's
`client.chat.completions.create()` with CSCS as `base_url` and your CSCS key as
`api_key`. Requests go to CSCS and run Apertus 8B. Model discovery uses
`client.models.list()`. See the [official SDK reference](https://developers.openai.com/api/reference/python).
It uses
temperature 0 and a bounded output length. The API still need not be perfectly
deterministic. JSON is requested through the prompt; undocumented server-side JSON
schema enforcement is not assumed. There is one API request per classification,
with automatic SDK retries explicitly disabled and no JSON repair calls.

The default prompt is `voting-nli-v2`: it requests short necessary citations and
explicitly checks negation, attribution and proposal scope. Set
`APERTUS_PROMPT_VERSION=voting-nli-v1` to reproduce the original prompt. Results and
evaluation configuration record the selected version. The output token budget
remains 768 to compare the prompt change independently.

An optional `APERTUS_PROMPT_VERSION=voting-nli-v3-ids` experiment asks the model to
select passage IDs only. Python attaches their exact full source text and metadata.
It marks these citations as `citation_scope: whole_passage` with
`quote_origin: local_source`; it does not claim that the model selected an exact
quote span. Existing quote validation still applies to v1/v2. See `EVALUATION.md`
for the experiment and its limits; ID validity does not imply semantic support.

The response parser requires an exact label and JSON shape. ENTAIL/CONTRADICT need
at least one supplied passage reference and a verifiable quote from that passage.
Matching tolerates whitespace differences, canonically equivalent Unicode accents,
and discretionary soft hyphens. A final fallback permits omitted source spaces
between letters (for example, `Substan zen` quoted as `Substanzen`); quote spaces
remain mandatory. It preserves letters, numbers, case, punctuation and visible
hyphens. These matches have `match_type: source_word_spacing` and
`requires_review: true`; the result also has `citation_review_required: true`.
Word joins can change meaning, so this flag must be considered when reviewing or
evaluating predictions. The model's label is retained, not independently verified.
Saved citations contain the exact original
source span, its character offsets, the model's quote and the matching method.
Citation pages and filenames are resolved locally, never trusted from model output.
These checks validate format and quote existence, not whether the evidence logically
supports the prediction. Human inspection and held-out evaluation remain necessary.

The prompt distinguishes evidence insufficiency from contradiction and preserves
qualifications, time periods and attribution of campaign arguments. It requests
NEUTRAL for missing or conflicting evidence, and no voting recommendations.

## Errors and limitations

- Missing document/collection: fix the path or document hash; this is an indexing
  error, not a factual NEUTRAL prediction.
- No retrieved evidence: returns NEUTRAL with `status: no_evidence` without calling
  Apertus. No model classification is claimed in that case.
- Authentication, quota/rate limits, timeouts, truncated replies, invalid labels,
  malformed JSON or invented citations: exit status 1 with `status: error` on stderr.
  Do not turn these failures into NEUTRAL labels in evaluation.
- With `--output`, failures are saved beside the requested result as
  `<stem>.error.json`, preserving any previous successful result. Citation validation
  failures include the raw model response and supplied evidence; quote mismatches
  also identify the rejected quote and its source passage. Inspect these before
  changing the prompt or matching rules.
- A local database lock means another process has it open; run indexing/search in
  sequence, and finish interrupted indexing before trusting the collection.
- `--top-k` defaults to 5 and is limited to 1–10 for this PoC. The prompt has a
  character-size guard which raises instead of silently truncating evidence. It
  is not an exact Apertus-token context check; context errors remain API errors.
- Hybrid mode requires a current sparse index; see `BM25.md` for rebuilding it.
- This has no calibrated relevance threshold and
  does not claim retrieval Recall@5 or classification accuracy. Similarity scores
  are not probabilities of factual correctness.

## Checkpoint

```sh
uv run python -m unittest discover -s tests -v
```

Offline tests use a mocked SDK boundary and cover prompt serialization, DE/FR/IT quote preservation, label/citation
validation, malformed and truncated responses, and error/NEUTRAL separation.
No live CSCS request or real-corpus classification was performed here: credentials,
the indexed booklets and the vector dependencies are not available in this workspace.

Next compare hybrid and dense-only evidence for a known claim, then measure
retrieval and classification separately on held-out examples.
