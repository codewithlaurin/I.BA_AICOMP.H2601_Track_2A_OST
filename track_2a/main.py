"""Judges' entrypoint: python main.py --input cases.jsonl --output predictions.jsonl."""

from scripts.benchmark import main

if __name__ == "__main__":
    raise SystemExit(main())
