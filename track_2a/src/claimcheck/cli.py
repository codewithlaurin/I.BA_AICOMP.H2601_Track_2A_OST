"""The judges' entrypoint: python -m claimcheck --input cases.jsonl --output predictions.jsonl

One JSON request per input line, one JSON prediction per output line, same ids. All input
lines are validated before anything is written; a bad input line exits 1 without output.
"""

import argparse
import json
import os
from pathlib import Path
import sys

from .contracts import ContractError, validate_prediction, validate_request
from .predict import ApertusPredictor, NeutralPredictor, Predictor


def read_requests(path: Path) -> list:
    requests, ids = [], set()
    with path.open(encoding="utf-8") as lines:
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                request = validate_request(json.loads(line), path.parent)
            except ValueError as exc:
                raise ContractError(f"{path}: line {number}: {exc}") from exc
            if request.id in ids:
                raise ContractError(f"{path}: line {number}: duplicate id {request.id!r}")
            ids.add(request.id)
            requests.append(request)
    if not requests:
        raise ContractError(f"{path}: input contains no requests")
    return requests


def run(input_path: Path, output_path: Path, predictor: Predictor) -> int:
    input_path, output_path = Path(input_path).resolve(), Path(output_path).resolve()
    if input_path == output_path:
        raise ContractError("--output must differ from --input")
    requests = read_requests(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for request in requests:
            prediction = validate_prediction(predictor.predict(request), request)
            out.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    return len(requests)


def main(argv: list[str] | None = None, *, predictor: Predictor | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify claims against Swiss voting booklets.")
    parser.add_argument("--input", required=True, type=Path, help="cases.jsonl")
    parser.add_argument("--output", required=True, type=Path, help="predictions.jsonl")
    args = parser.parse_args(argv)
    if predictor is None:
        if any(os.environ.get(name, "").strip() for name in ("API_KEY", "LLM_API_KEY", "CSCS_INFERENCE_API_KEY")):
            predictor = ApertusPredictor()
        else:
            print("claimcheck: API_KEY not set; using neutral stub, no inference", file=sys.stderr)
            predictor = NeutralPredictor()
    try:
        count = run(args.input, args.output, predictor)
    except Exception as exc:
        print(f"claimcheck: error: {exc}", file=sys.stderr)
        return 1
    print(f"claimcheck: wrote {count} predictions to {args.output}", file=sys.stderr)
    return 0
