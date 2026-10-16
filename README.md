# Apertus ClaimCheck

Multilingual claim verification against official Swiss voting booklets, built for
**Hack Apertus, Track 2A (OST)** by **HSLU Team 1: Karim Darwiche and Laurin Lötscher**.

Given a claim and source material, ClaimCheck uses Apertus v1.5 to classify the claim
as **entailment (0)**, **neutral (1)**, or **contradiction (2)**. It supports German,
French, and Italian, including claims and sources in different languages, and
returns source evidence with token and timing metrics.

- **Task A:** retrieve relevant pages from a voting booklet PDF, then classify the
  claim and return evidence with PDF page numbers.
- **Task B:** classify the claim against a supplied reference passage.

The application checks claims against the supplied source material; predictions
are not political recommendations.

## Quick start

Requirements: Docker running locally and an API key for an OpenAI-compatible
Apertus endpoint. The image targets `linux/amd64` and runs on CPU; the build needs
internet access to install dependencies and include the retrieval model.

From the repository root:

```bash
cd track_2a
export BASE_URL=https://api.inference.cscs.ch/v1
export API_KEY="your-api-key"
export LLM_NAME=swiss-ai/Apertus-v1.5-70B
make run
```

Replace `your-api-key` with your runtime credential. A missing or blank API key
causes a configuration error before predictions are written.

This builds the image, reads the included `data/cases.jsonl` sample, and writes
`output/predictions.jsonl`. Inputs are mounted read-only, and credentials are
passed at runtime. To use your own cases:

```bash
make run INPUT=/path/to/cases.jsonl OUTPUT=/path/to/predictions.jsonl
```

Place the referenced PDFs in `booklets/` next to the cases file. See the
[setup guide](track_2a/README.md) for the input format, Docker commands, and local
scoring instructions.

## Submission container

The published `linux/amd64` image is available from
[GitHub Container Registry](https://github.com/users/codewithlaurin/packages/container/package/i.ba_aicomp.h2601_track_2a_ost):

```bash
docker pull --platform linux/amd64 ghcr.io/codewithlaurin/i.ba_aicomp.h2601_track_2a_ost:latest
```

Immutable image reference:

```text
ghcr.io/codewithlaurin/i.ba_aicomp.h2601_track_2a_ost@sha256:f2faee4de008296d6ecdc50b5393b1d75a24baa0225cd9211085b449a4c1045c
```

Run it from `track_2a/` with runtime credentials set as above:

```bash
mkdir -p output
docker run --rm --platform linux/amd64 \
  -e BASE_URL -e API_KEY -e LLM_NAME \
  -v "$PWD/data/cases.jsonl:/data/cases.jsonl:ro" \
  -v "$PWD/data/booklets:/data/booklets:ro" \
  -v "$PWD/output:/output" \
  ghcr.io/codewithlaurin/i.ba_aicomp.h2601_track_2a_ost:latest \
  --input /data/cases.jsonl --output /output/predictions.jsonl
```

## Documentation

- [Setup and execution](track_2a/README.md)
- [Technical report](track_2a/technical_report.md): architecture, evaluation,
  limitations, and reproducibility
- [Recorded development results](track_2a/docs/RESULTS.md): historical experiments
  and their configurations
- [Implementation](track_2a/src/claimcheck/)

## Demo

Demo link forthcoming. A local browser demo is available in
[scripts/demo_gui.py](track_2a/scripts/demo_gui.py); its launch command is documented
in the [technical report](track_2a/technical_report.md#72-run-and-score).

## License

The software is licensed under [Apache License 2.0](LICENSE). Third-party data,
documents, models, and dependencies retain their respective licenses.
