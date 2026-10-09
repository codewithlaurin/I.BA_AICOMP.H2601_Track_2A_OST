"""JSONL command-line adapter with atomic output and contextual diagnostics."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from .contracts import Booklet, ContractError, Request, validate_prediction, validate_request
from .predict import NeutralPredictor, Predictor


def _reject_constant(value: str):
    raise ContractError(f"invalid JSON number {value}")


def _unique_keys(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def read_requests(path: Path) -> list[tuple[int, Request]]:
    requests = []
    seen = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                if not line.strip():
                    raise ContractError("blank line; expected one JSON object per line")
                obj = json.loads(line, parse_constant=_reject_constant, object_pairs_hook=_unique_keys)
                request = validate_request(obj, path.resolve().parent)
                if request.id in seen:
                    raise ContractError(
                        f"duplicate id {request.id!r}; first seen on line {seen[request.id]}"
                    )
                seen[request.id] = line_number
                requests.append((line_number, request))
            except (ValueError, OSError) as exc:
                raise ContractError(f"{path}: line {line_number}: {exc}") from exc
    if not requests:
        raise ContractError(f"{path}: input contains no requests")
    return requests


def _same_file(left: Path, right: Path) -> bool:
    return left == right or (left.exists() and right.exists() and left.samefile(right))


def run(input_path: Path, output_path: Path, predictor: Predictor) -> int:
    """Validate the full input, then publish output only if every case succeeds."""
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    if _same_file(input_path, output_path):
        raise ContractError("--output must differ from --input")
    requests = read_requests(input_path)
    for _, request in requests:
        if isinstance(request.source, Booklet) and _same_file(output_path, request.source.path):
            raise ContractError(f"--output must not overwrite booklet {request.source.path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_path.parent,
            prefix=f".{output_path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            for line_number, request in requests:
                try:
                    prediction = validate_prediction(predictor.predict(request), request)
                    stream.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n")
                except Exception as exc:
                    raise ContractError(
                        f"{input_path}: line {line_number}, id {request.id!r}: {exc}"
                    ) from exc
        os.replace(temporary, output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return len(requests)


def main(argv: list[str] | None = None, *, predictor: Predictor | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate cases and write claimcheck predictions.")
    parser.add_argument("--input", required=True, type=Path, help="UTF-8 cases JSONL file")
    parser.add_argument("--output", required=True, type=Path, help="Predictions JSONL destination")
    args = parser.parse_args(argv)
    if predictor is None:
        print("claimcheck: using neutral infrastructure stub; no inference or PDF extraction", file=sys.stderr)
        predictor = NeutralPredictor()
    try:
        count = run(args.input, args.output, predictor)
    except Exception as exc:
        print(f"claimcheck: error: {exc}", file=sys.stderr)
        return 1
    print(f"claimcheck: wrote {count} predictions to {args.output}", file=sys.stderr)
    return 0
