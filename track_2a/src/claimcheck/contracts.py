"""Request and prediction contracts, independent of inference and PDF parsing."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

LANGUAGES = frozenset({"de", "fr", "it"})
LABEL_NAMES = {0: "entailment", 1: "neutral", 2: "contradiction"}


class ContractError(ValueError):
    """An input or prediction does not satisfy the public contract."""


def _object(value: Any, field: str) -> dict:
    if not isinstance(value, dict):
        raise ContractError(f"{field} must be an object")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field} must be a non-empty string")
    return value


def _language(value: Any, field: str) -> str:
    if not isinstance(value, str) or value not in LANGUAGES:
        raise ContractError(f"{field} must be one of de, fr, it")
    return value


@dataclass(frozen=True)
class TextSource:
    text: str
    language: str


@dataclass(frozen=True)
class Booklet:
    path: Path
    language: str


@dataclass(frozen=True)
class Request:
    id: str | int
    vote: Any
    claim: TextSource
    source: Booklet | TextSource

    @property
    def task(self) -> str:
        return "A" if isinstance(self.source, Booklet) else "B"


def validate_request(value: Any, input_dir: Path) -> Request:
    """Resolve booklet paths relative to the JSONL file, never the working dir."""
    obj = _object(value, "request")
    for field in ("id", "vote", "claim"):
        if field not in obj:
            raise ContractError(f"missing required field {field!r}")
    identifier = obj["id"]
    if type(identifier) not in (str, int) or (
        isinstance(identifier, str) and not identifier.strip()
    ):
        raise ContractError("id must be a non-empty string or an integer")
    if ("booklet" in obj) == ("reference" in obj):
        raise ContractError("request must contain exactly one of booklet or reference")

    claim = _object(obj["claim"], "claim")
    claim = TextSource(
        _text(claim.get("text"), "claim.text"),
        _language(claim.get("language"), "claim.language"),
    )
    if "booklet" in obj:
        booklet = _object(obj["booklet"], "booklet")
        path_text = _text(booklet.get("path"), "booklet.path")
        if "\x00" in path_text:
            raise ContractError("booklet.path must not contain a null character")
        path = Path(path_text)
        if not path.is_absolute():
            path = input_dir / path
        source = Booklet(
            path.resolve(), _language(booklet.get("language"), "booklet.language")
        )
    else:
        reference = _object(obj["reference"], "reference")
        source = TextSource(
            _text(reference.get("text"), "reference.text"),
            _language(reference.get("language"), "reference.language"),
        )
    return Request(identifier, obj["vote"], claim, source)


def validate_prediction(value: Any, request: Request) -> dict:
    """Validate a predictor's output before it can reach the JSONL file."""
    obj = _object(value, "prediction")
    identifier = obj.get("id")
    if type(identifier) is not type(request.id) or identifier != request.id:
        raise ContractError("prediction.id must match the input id unchanged")
    label = obj.get("label")
    if type(label) is not int or label not in LABEL_NAMES:
        raise ContractError("prediction.label must be 0, 1, or 2")
    if obj.get("label_name") != LABEL_NAMES[label]:
        raise ContractError(f"label {label} requires label_name {LABEL_NAMES[label]!r}")
    evidence = obj.get("evidence")
    if not isinstance(evidence, list) or len(evidence) > 5:
        raise ContractError("prediction.evidence must be a list with at most five items")
    if request.task == "A" and label in (0, 2) and not evidence:
        raise ContractError("Task A labels 0 and 2 require evidence")
    checked_evidence = []
    for index, item in enumerate(evidence):
        field = f"evidence[{index}]"
        item = _object(item, field)
        if "page" not in item:
            raise ContractError(f"{field}.page is required (null for Task B)")
        page = item["page"]
        if request.task == "A" and (type(page) is not int or page < 1):
            raise ContractError(f"{field}.page must be a 1-based PDF page number")
        if request.task == "B" and page is not None:
            raise ContractError(f"{field}.page must be null for Task B")
        text = _text(item.get("text"), f"{field}.text")
        if len(text) > 5000:
            raise ContractError(f"{field}.text exceeds the 5,000-character limit")
        checked_evidence.append({"page": page, "text": text})
    metrics = _object(obj.get("metrics"), "prediction.metrics")
    for field in ("input_tokens", "output_tokens"):
        if type(metrics.get(field)) is not int or metrics[field] < 0:
            raise ContractError(f"metrics.{field} must be a non-negative integer")
    elapsed = metrics.get("inference_time_ms")
    if type(elapsed) not in (int, float) or elapsed < 0 or (
        isinstance(elapsed, float) and not math.isfinite(elapsed)
    ):
        raise ContractError("metrics.inference_time_ms must be a finite non-negative number")
    return {
        "id": identifier,
        "label": label,
        "label_name": LABEL_NAMES[label],
        "evidence": checked_evidence,
        "metrics": {field: metrics[field] for field in (
            "input_tokens", "output_tokens", "inference_time_ms"
        )},
    }
