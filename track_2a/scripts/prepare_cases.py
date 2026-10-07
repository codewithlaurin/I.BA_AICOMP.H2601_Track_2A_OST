"""Convert a local public dataset export to judge cases and separate gold labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from urllib.request import Request, urlopen

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.benchmark import validate_case


def prepare(input_path: Path, output_dir: Path, task: str, limit: int | None,
            seed: int, download_booklets: bool) -> int:
    raw = input_path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    selected = list(enumerate(rows))
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        selected = sorted(random.Random(seed).sample(selected, min(limit, len(selected))))
    cases, expected, downloads, ids = [], [], {}, set()
    for position, row in selected:
        index = row.get("row_index", position)
        label = row["entailment_label"]
        if type(index) is not int or index < 0 or type(label) is not int or label not in range(3):
            raise ValueError("Expected a nonnegative row index and integer label 0, 1 or 2")
        for current_task in (("A", "B") if task == "both" else (task,)):
            case = {"id": f"v1.0-row-{index}-{current_task}", "vote": row["vote"],
                    "claim": {"text": row["claim"], "language": row["claim_language"]}}
            if current_task == "B":
                case["reference"] = {"text": row["reference_string"], "language": row["reference_language"]}
            else:
                url = row["booklet_url"]
                filename = "booklets/" + hashlib.sha256(url.encode()).hexdigest()[:24] + ".pdf"
                case["booklet"] = {"path": filename, "language": row["reference_language"]}
                downloads[filename] = url
            validate_case(case)
            if case["id"] in ids:
                raise ValueError("Duplicate original row indices")
            ids.add(case["id"])
            cases.append(case)
            expected.append({"id": case["id"], "label": label})
    output_dir.mkdir(parents=True, exist_ok=False)
    if download_booklets:
        for filename, url in downloads.items():
            if not url.startswith("https://"):
                raise ValueError("Booklet downloads require an HTTPS URL")
            with urlopen(Request(url, headers={"User-Agent": "HackApertus-evaluation/1.0"}), timeout=60) as response:
                pdf = response.read()
            if not pdf.startswith(b"%PDF"):
                raise ValueError(f"Download did not return a PDF: {url}")
            path = output_dir / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pdf)
    for filename, records in (("cases.jsonl", cases), ("expected-labels.jsonl", expected)):
        (output_dir / filename).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    (output_dir / "preparation.json").write_text(json.dumps({
        "dataset_sha256": hashlib.sha256(raw).hexdigest(), "seed": seed, "task": task,
        "selected_row_indices": [row.get("row_index", position) for position, row in selected],
        "booklet_urls": downloads, "booklets_downloaded": download_booklets,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(cases)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Dataset export, e.g. v1.0.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True, help="Must be a new directory")
    parser.add_argument("--task", choices=["A", "B", "both"], default="both")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download-booklets", action="store_true")
    args = parser.parse_args()
    try:
        count = prepare(args.input, args.output_dir, args.task, args.limit, args.seed, args.download_booklets)
        print(f"Wrote {count} cases and separate gold labels to {args.output_dir}")
        return 0
    except Exception as error:
        print(f"Preparation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
