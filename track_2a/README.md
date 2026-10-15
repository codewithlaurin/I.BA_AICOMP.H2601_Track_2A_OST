# Academia Challenges

Submissions must use the Apertus model family.
For Track 2 this means that submitted solutions must be built with Apertus. Other open-weights models can be used to support development, e.g. as automatic judges during evaluation. Their role must be clearly described in the submission report.

💬 In case you have questions, join the conversation on Discord or send an email to “hello@hackapertus.ch”

## How it works
Pick from 5 academia challenges provided by Swiss academic institutions:

- **FHGR:** AI-Powered Job Interview Coach
- **OpenParlData:** Extracting Parliamentary Affairs from PDFs into One Common Structure
- **OST:** Multilingual Natural Language Inference over Swiss Official Voting Booklets
- **UZH:** Detecting Cross-Lingual Semantic Differences in Swiss Government Websites
- **ZHAW:** See It, Say It, Pick It: Vision-Language Grounding for a Real Robot Arm

The challenges incl. submission and judging criteria are described in our **Getting Started guide**:
https://hackapertus.notion.site/getting-started-guide-onlinehack

## Run it

Keep `track_2a/` as it is: don't rename it or move its files, just delete the
other track directories.

From `track_2a/`:

```bash
export BASE_URL=https://api.inference.cscs.ch/v1 API_KEY=... LLM_NAME=swiss-ai/Apertus-v1.5-8B
make run                                     # data/cases.jsonl -> output/predictions.jsonl
make run INPUT=/path/cases.jsonl OUTPUT=/path/predictions.jsonl
```

`make run` builds the `linux/amd64` image ([Dockerfile](Dockerfile)) and runs the CLI in it.
Only the cases file and the `booklets/` folder next to it are mounted (booklet paths in the
cases file are relative to it); predictions are written next to `OUTPUT`. A local `.env` is
passed to the container if present. The same image without make:

```bash
docker build --platform linux/amd64 -t apertus-claimcheck .
docker run --rm --platform linux/amd64 -e BASE_URL -e API_KEY -e LLM_NAME \
  -v "$PWD/data/cases.jsonl:/data/cases.jsonl:ro" -v "$PWD/data/booklets:/data/booklets:ro" \
  -v "$PWD/output:/output" \
  apertus-claimcheck --input /data/cases.jsonl --output /output/predictions.jsonl
```

Requirements:
- **Runtime:** Docker (image is `linux/amd64`); the build needs internet, the run only reaches the Apertus endpoint.
- **Hardware:** CPU only, no GPU. Peak memory 1.1 GB on 50 Task A cases (29 booklets cached); image about 3 GB.
- **API keys:** `API_KEY` for the OpenAI-compatible Apertus endpoint (`BASE_URL`, default CSCS); `LLM_NAME` defaults to `swiss-ai/Apertus-v1.5-8B`. No key is stored in the image.
- **Model weights:** Apertus runs remotely. The page-retrieval embedder `intfloat/multilingual-e5-small` (CPU) is downloaded at build time and baked into the image; the container runs with `HF_HUB_OFFLINE=1`.

Scoring locally with the organisers' script:

```bash
uv run --with scikit-learn python scripts/official_evaluate.py \
  --predictions output/predictions.jsonl --expected expected-labels.jsonl --cases cases.jsonl
```

## Data
The `data/` directory must not exceed 100 MB.


## 📦 Submission Requirements & Deliverables
❗️ Submissions are not handled on Devpost. Submit through our website only:
http://hackapertus.ch/online-hack/submissions

Requirements differ by challenge. See the description of the challenge you are entering for the exact deliverables.


## ⚖️ Judging Criteria
Judging criteria also differ by challenge. See the respective challenge description.


## Support

**Licensing requirements**
Please check our Terms & Conditions (6. What you build is open source):
https://hackapertus.ch/terms-and-conditions

## FAQ
💡 https://hackapertus.ch/faq

## Contact
💬 In case you have questions, join the conversation on Discord or send an email to “hello@hackapertus.ch”
