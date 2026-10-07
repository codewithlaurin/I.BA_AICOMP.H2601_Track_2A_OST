"""Score judges' predictions against separate gold labels, without any model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.benchmark import (LABELS, booklet_path, extract_pdf, read_jsonl,
                               validate_case, validate_response)


def metrics_for(rows: list[dict]) -> dict:
    matrix = [[0] * 4 for _ in range(3)]
    for row in rows:
        matrix[row["gold"]][row["predicted"] if row["predicted"] is not None else 3] += 1
    f1 = []
    for label in range(3):
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in range(3) if other != label)
        fn = sum(matrix[label]) - tp
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    valid = sum(row["predicted"] is not None for row in rows)
    correct = sum(matrix[label][label] for label in range(3))
    usage = {}
    for key in ("input_tokens", "output_tokens", "inference_time_ms"):
        values = [row.get("usage", {}).get(key) for row in rows]
        known = [v for v in values if type(v) in (int, float) and math.isfinite(v) and v >= 0]
        usage[key] = {"reported_total": sum(known), "missing_cases": len(rows) - len(known)}
    return {
        "cases": len(rows), "valid_predictions": valid, "errors": len(rows) - valid,
        "accuracy": correct / len(rows) if rows else None,
        "prediction_coverage": valid / len(rows) if rows else None,
        "macro_f1": sum(f1) / 3 if rows else None,
        "f1_per_label": dict(zip(LABELS, f1)),
        "confusion_matrix": {"rows": list(LABELS), "columns": [*LABELS, "ERROR"], "values": matrix},
        "usage": usage,
    }


def score(cases: list[dict], predictions: list[dict], expected: list[dict],
          *, data_root: Path | None = None) -> dict:
    case_ids = {case["id"] for case in cases}
    gold = {row["id"]: row.get("label") for row in expected}
    predicted = {row["id"]: row for row in predictions}
    if len(case_ids) != len(cases) or len(gold) != len(expected) or len(predicted) != len(predictions):
        raise ValueError("Duplicate IDs are not allowed")
    if set(gold) != case_ids or set(predicted) - case_ids:
        raise ValueError("Gold IDs must exactly match cases; predictions must not contain unknown IDs")
    if any(type(label) is not int or label not in range(3) for label in gold.values()):
        raise ValueError("Gold labels must be integers 0, 1 or 2")
    rows, errors = [], []
    for case in cases:
        task = validate_case(case)
        result = predicted.get(case["id"])
        label = None
        try:
            if result is None:
                raise ValueError("Missing prediction")
            if "error" in result:
                raise ValueError(f"Prediction failed: {result['error']}")
            validate_response(case, result)
            if task == "A" and data_root is not None:
                pages = dict(extract_pdf(booklet_path(case, data_root)))
                for citation in result["evidence"]:
                    text = pages.get(citation["page"], "")
                    if citation["text"] not in text or citation["text"].strip() == text.strip():
                        raise ValueError("Task A evidence is not a verbatim excerpt of the cited PDF page")
            label = result["label"]
        except (ValueError, RuntimeError) as error:
            errors.append({"id": case["id"], "error": str(error)})
        source = case.get("reference", case.get("booklet"))
        usage = result.get("metrics", {}) if result else {}
        rows.append({"gold": gold[case["id"]], "predicted": label,
                     "task": task, "claim_language": case["claim"]["language"],
                     "language_pair": f"{source['language']}->{case['claim']['language']}",
                     "usage": usage if isinstance(usage, dict) else {}})
    tasks = {}
    for task in ("A", "B"):
        group = [row for row in rows if row["task"] == task]
        tasks[task] = metrics_for(group)
        for field in ("claim_language", "language_pair"):
            tasks[task][f"by_{field}"] = {
                language: metrics_for([row for row in group if row[field] == language])
                for language in sorted({row[field] for row in group})}
    return {"overall": metrics_for(rows), "by_task": tasks, "invalid_predictions": errors,
            "pdf_quotes_verified": data_root is not None,
            "evidence_note": "Verbatim checks do not measure evidence relevance or semantic correctness."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Judge-format cases.jsonl")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True, help="Separate expected-labels.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Summary JSON")
    parser.add_argument("--verify-pdfs", action="store_true", help="Also check Task A quotes against PDF text")
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    try:
        inputs = {"cases": args.input, "predictions": args.predictions, "expected": args.expected}
        if args.output.resolve() in {path.resolve() for path in inputs.values()}:
            raise ValueError("Summary output must not overwrite an input file")
        summary = score(read_jsonl(args.input), read_jsonl(args.predictions), read_jsonl(args.expected),
                        data_root=(args.data_root or args.input.parent) if args.verify_pdfs else None)
        summary["input_sha256"] = {
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in inputs.items()}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for task, metrics in summary["by_task"].items():
            print(f"Task {task}: cases={metrics['cases']}, macro-F1={metrics['macro_f1']}, "
                  f"errors={metrics['errors']}")
        return 1 if summary["invalid_predictions"] else 0
    except Exception as error:
        print(f"Scoring failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
