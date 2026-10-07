"""Full-document/reference baseline implementing the OST judges' JSONL contract."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from functools import lru_cache
from pathlib import Path
from time import perf_counter

from .apertus_nli import ApiConfig, InferenceError, SYSTEM_PROMPT, api_request, parse_prediction

LABELS = ("entailment", "neutral", "contradiction")
LABEL_IDS = {"ENTAIL": 0, "NEUTRAL": 1, "CONTRADICT": 2}
LANGUAGES = {"de", "fr", "it"}
DEFAULT_MODEL = "swiss-ai/Apertus-v1.5-8B"
MAX_QUOTE_CHARS = 500
BENCHMARK_PROMPT = SYSTEM_PROMPT + """
The vote field identifies the proposal being checked. In a booklet containing
multiple proposals, use evidence about that proposal. The source and claim may
have different languages. A title alone supplies no unstated details.
Keep the explanation to one sentence. Cite only the minimum necessary short,
contiguous source quotes (at most 500 characters each), in the source language.
Never cite an entire page or an entire reference passage. Do not translate quotes.
"""


def read_jsonl(path: Path) -> list[dict]:
    records, ids = [], set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            identifier = record.get("id")
            if not isinstance(identifier, str) or not identifier.strip() or identifier in ids:
                raise ValueError(f"{path}:{line_number}: missing, invalid or duplicate id")
            ids.add(identifier)
            records.append(record)
    return records


def require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def validate_case(case: dict) -> str:
    require_text(case.get("id"), "id")
    require_text(case.get("vote"), "vote")
    if ("booklet" in case) == ("reference" in case):
        raise ValueError("Provide exactly one of booklet or reference")
    source = "booklet" if "booklet" in case else "reference"
    for name in ("claim", source):
        value = case.get(name)
        if not isinstance(value, dict) or value.get("language") not in LANGUAGES:
            raise ValueError(f"{name} must have language de, fr or it")
        require_text(value.get("path" if name == "booklet" else "text"), name)
    return "A" if source == "booklet" else "B"


def booklet_path(case: dict, data_root: Path) -> Path:
    relative = Path(case["booklet"]["path"])
    root = data_root.resolve()
    resolved = (root / relative).resolve()
    if relative.is_absolute() or not resolved.is_relative_to(root):
        raise ValueError("booklet.path must be relative and stay within the input data directory")
    if not resolved.is_file():
        raise ValueError(f"Booklet PDF is missing: {relative}")
    return resolved


@lru_cache(maxsize=16)
def extract_pdf(path: Path) -> tuple[tuple[int, str], ...]:
    """CPU text extraction, cached only by source path; never reads evaluation labels."""
    import pypdfium2 as pdfium

    pages = []
    with pdfium.PdfDocument(path) as document:
        for index in range(len(document)):
            page = document[index]
            try:
                textpage = page.get_textpage()
                try:
                    text = textpage.get_text_bounded()
                finally:
                    textpage.close()
            finally:
                page.close()
            if text.strip():
                pages.append((index + 1, text))
    if not pages:
        raise ValueError("PDF contains no extractable text; this baseline does not perform OCR")
    return tuple(pages)


def evidence_for(case: dict, data_root: Path) -> list[dict]:
    if "reference" in case:
        source = case["reference"]
        pages = ((None, source["text"]),)
        source_pdf = None
    else:
        source = case["booklet"]
        path = booklet_path(case, data_root)
        pages = extract_pdf(path)
        source_pdf = source["path"]
    # IDs and provenance are local to this request; no prebuilt database is needed.
    return [{"evidence_id": f"E{index}", "page_number": page, "text": text,
             "language": source["language"], "source_pdf": source_pdf,
             "document_id": case["id"], "chunk_id": f"{case['id']}:{index}"}
            for index, (page, text) in enumerate(pages, 1)]


def benchmark_config() -> ApiConfig:
    legacy = ApiConfig.from_env()
    model = os.environ.get("LLM_NAME", os.environ.get("APERTUS_MODEL", DEFAULT_MODEL))
    if not model.startswith("swiss-ai/Apertus-v1.5-"):
        raise ValueError("Judging requires Apertus v1.5; set LLM_NAME=swiss-ai/Apertus-v1.5-8B")
    key = os.environ.get("API_KEY", os.environ.get("LLM_API_KEY",
        os.environ.get("CSCS_INFERENCE_API_KEY", ""))).strip()
    if not key or not legacy.base_url.strip():
        raise ValueError("Set BASE_URL and API_KEY (legacy CSCS variables are also supported)")
    return ApiConfig(base_url=legacy.base_url, model=model, max_output_tokens=1024,
                     max_prompt_chars=300_000)


def token_usage(response: dict) -> dict:
    usage = response.get("usage") or {}
    result = {"input_tokens": usage.get("prompt_tokens"),
              "output_tokens": usage.get("completion_tokens")}
    # Chat Completions completion_tokens already includes reasoning tokens.
    for key, value in result.items():
        if type(value) is not int or value < 0:
            raise InferenceError(f"API response has no valid {key}; cannot report actual usage")
    return result


def validate_response(case: dict, response: dict) -> None:
    """Validate the public response shape (also used by the local scorer)."""
    label = response.get("label")
    if response.get("id") != case["id"] or type(label) is not int or label not in range(3):
        raise ValueError("Invalid response id or label")
    if response.get("label_name") != LABELS[label]:
        raise ValueError("label_name does not match label")
    evidence = response.get("evidence")
    if not isinstance(evidence, list) or (label != 1 and not evidence):
        raise ValueError("Entailment and contradiction require evidence")
    for quote in evidence:
        if not isinstance(quote, dict):
            raise ValueError("Evidence must contain objects")
        text = require_text(quote.get("text"), "evidence.text")
        if len(text) > MAX_QUOTE_CHARS or "page" not in quote:
            raise ValueError("Evidence must be a short quote with a page field")
        page = quote["page"]
        if "reference" in case:
            reference = case["reference"]["text"]
            if page is not None or text not in reference or text.strip() == reference.strip():
                raise ValueError("Task B needs a verbatim excerpt of the reference and page=null")
        elif type(page) is not int or page < 1:
            raise ValueError("Task A needs a 1-based PDF page number")
    metrics = response.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Missing usage metrics")
    for key in ("input_tokens", "output_tokens"):
        if type(metrics.get(key)) is not int or metrics[key] < 0:
            raise ValueError(f"Invalid metrics.{key}")
    time = metrics.get("inference_time_ms")
    if type(time) not in (int, float) or not math.isfinite(time) or time < 0:
        raise ValueError("Invalid inference_time_ms")


def predict(case: dict, data_root: Path, config: ApiConfig) -> dict:
    start = perf_counter()
    metrics = {"input_tokens": 0, "output_tokens": 0}
    try:
        evidence = evidence_for(case, data_root)
        data = {"vote": case["vote"], "claim": case["claim"],
                "evidence": [{key: item[key] for key in
                              ("evidence_id", "page_number", "language", "text")}
                             for item in evidence]}
        content = json.dumps(data, ensure_ascii=False)
        if len(content) + len(BENCHMARK_PROMPT) > config.max_prompt_chars:
            raise ValueError("Full document exceeds the baseline prompt budget; no text was truncated")
        # Exactly one call per case. Unknown usage stays unknown on transport failures.
        metrics = {"input_tokens": None, "output_tokens": None}
        raw = api_request("chat/completions", config, {
            "model": config.model, "temperature": 0, "max_tokens": config.max_output_tokens,
            "stream": False,
            "messages": [{"role": "system", "content": BENCHMARK_PROMPT},
                         {"role": "user", "content": content}],
        })
        metrics = token_usage(raw)
        returned_model = raw.get("model", config.model)
        if not returned_model.startswith("swiss-ai/Apertus-v1.5-"):
            raise InferenceError("Endpoint returned a model outside the required Apertus v1.5 family")
        choice = raw["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise InferenceError(f"Incomplete model response: {choice.get('finish_reason')}")
        parsed = parse_prediction(choice["message"]["content"], evidence)
        by_id = {item["evidence_id"]: item for item in evidence}
        citations = []
        for citation in parsed["citations"]:
            source = by_id[citation["evidence_id"]]
            if citation["quote"].strip() == source["text"].strip():
                raise InferenceError("A whole source passage is not a short evidence excerpt")
            citations.append({"page": citation["page_number"], "text": citation["quote"]})
        label = LABEL_IDS[parsed["label"]]
        metrics["inference_time_ms"] = round((perf_counter() - start) * 1000, 3)
        prediction = {"id": case["id"], "label": label, "label_name": LABELS[label],
                      "evidence": citations, "metrics": metrics}
        validate_response(case, prediction)
        return prediction
    except Exception as error:
        metrics["inference_time_ms"] = round((perf_counter() - start) * 1000, 3)
        # Explicit invalid response, counted as wrong by the scorer/judges. Never
        # turn a technical failure into a fabricated neutral prediction or usage.
        result = {"id": case["id"], "error": str(error), "metrics": metrics}
        if isinstance(error, InferenceError) and error.details:
            result["diagnostics"] = error.details
        return result


def run(input_path: Path, output_path: Path, data_root: Path | None = None) -> int:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different files")
    cases = read_jsonl(input_path)
    root = (data_root or input_path.parent).resolve()
    for case in cases:
        validate_case(case)
        if "booklet" in case:
            path = booklet_path(case, root)
            if path == output_path.resolve():
                raise ValueError("Output must not overwrite an input booklet")
    config = benchmark_config() if cases else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with output_path.open("w", encoding="utf-8") as output:
        for case in cases:
            result = predict(case, root, config)
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            if "error" in result:
                failures += 1
                print(f"{case['id']}: {result['error']}", file=sys.stderr)
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path,
                        help="PDF path base; defaults to the input file's directory (/data for judges)")
    args = parser.parse_args()
    try:
        return run(args.input, args.output, args.data_root)
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1
