"""Evaluate retrieval (Recall@k) and zero-shot NLI classification on the OST dataset.

Two separate numbers, because they isolate two separate failure modes:

* Recall@k   – is the gold ``reference_string`` contained in one of the k retrieved
               chunks? Needs no API call. Rows without a reference (mostly NEUTRAL)
               are excluded from this metric.
* Accuracy / macro-F1 – does Apertus return the gold label? One API call per claim,
               cached under a hash of (claim, evidence, prompt version, model) so that
               re-runs with an unchanged setup are free.

Typical use, from ``track_2a``::

    uv run python scripts/evaluate.py --export                       # once: HF -> data/evaluation/train.jsonl
    uv run python scripts/evaluate.py --limit 30 --seed 42 --retrieval-only --run-name retrieval-smoke
    uv run --env-file .env python scripts/evaluate.py --limit 30 --seed 42 --run-name hybrid-smoke
    uv run --env-file .env python scripts/evaluate.py --run-name hybrid-full

Each run writes ``data/evaluation/runs/<run-name>/{predictions.jsonl,metrics.json,metrics.md}``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from apertus_nli import PROMPT_VERSION, ApiConfig, InferenceError, classify_claim
from booklet_manifest import attach_document_metadata, load_document_metadata
from check_claim import retrieve_evidence

DATASET = "OSTswiss/MNLIoverSwissVotingBooklets"
DATASETS_SERVER = "https://datasets-server.huggingface.co"
LABELS = ("ENTAIL", "NEUTRAL", "CONTRADICT")
# Confirmed by Laurin (see docs/EVALUATION.md). Override with --label-map if the dataset changes.
DEFAULT_LABEL_MAP = {"0": "ENTAIL", "1": "NEUTRAL", "2": "CONTRADICT"}
FUZZY_THRESHOLD = 85   # partial-ratio threshold for chunk/reference overlap
MIN_CHUNK_CHARS = 40   # ignore tiny chunks (page numbers, headers) when matching


# --------------------------------------------------------------------------- dataset

def export_dataset(path: Path, config: str = "default", split: str = "train") -> int:
    """Download the full split through the Dataset Viewer API (no `datasets` dependency)."""
    if path.exists():
        raise SystemExit(f"{path} exists; delete it first if you really want to re-export")
    path.parent.mkdir(parents=True, exist_ok=True)
    offset, total, written = 0, None, 0
    with path.open("w", encoding="utf-8") as out:
        while total is None or offset < total:
            params = urlencode({"dataset": DATASET, "config": config, "split": split,
                                "offset": offset, "length": 100})
            request = Request(f"{DATASETS_SERVER}/rows?{params}",
                              headers={"User-Agent": "ApertusEvaluator/1.0"})
            with urlopen(request, timeout=90) as response:
                payload = json.loads(response.read())
            total = payload["num_rows_total"]
            for item in payload["rows"]:
                row = {**item["row"], "config": config, "split": split, "row_index": item["row_idx"]}
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            offset += len(payload["rows"])
            if not payload["rows"]:
                break
    print(f"Saved {written} rows to {path}")
    return written


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def gold_label(row: dict, label_map: dict[str, str], field: str) -> str:
    raw = row[field]
    label = label_map.get(str(raw), str(raw)).upper()
    if label not in LABELS:
        raise ValueError(f"Row {row.get('row_index')}: unknown label {raw!r}")
    return label


def stratified_sample(rows: list[dict], limit: int, seed: int, key) -> list[dict]:
    """Round-robin over (language, label) buckets so small samples stay balanced."""
    if limit >= len(rows):
        return rows
    rng = random.Random(seed)
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[key(row)].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    picked: list[dict] = []
    while len(picked) < limit:
        progressed = False
        for bucket in buckets.values():
            if bucket and len(picked) < limit:
                picked.append(bucket.pop())
                progressed = True
        if not progressed:
            break
    return picked


# --------------------------------------------------------------------------- manifest

def url_to_document_id(manifest_path: Path) -> dict[str, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mapping = {}
    for item in manifest["files"]:
        if item.get("sha256") and item.get("url"):
            mapping[item["url"].strip()] = item["sha256"]
    if not mapping:
        raise SystemExit("Manifest has no url/sha256 entries; run download_booklets.py first")
    return mapping


# --------------------------------------------------------------------------- matching

def normalize(text: str) -> str:
    """Lowercase, NFC, drop soft hyphens, join hyphenated line breaks, collapse whitespace."""
    text = unicodedata.normalize("NFC", text).replace("­", "")
    text = re.sub(r"(\w)[-‐‑]\s*\n\s*(\w)", r"\1\2", text)   # Silben-\ntrennung
    text = re.sub(r"[\"'«»‹›„“”‘’]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _partial_ratio(needle: str, haystack: str) -> float:
    try:
        from rapidfuzz import fuzz
        return float(fuzz.partial_ratio(needle, haystack))
    except ImportError:
        from difflib import SequenceMatcher
        if len(needle) >= len(haystack):
            return SequenceMatcher(None, needle, haystack).ratio() * 100
        best = 0.0
        step = max(1, len(needle) // 4)
        for start in range(0, len(haystack) - len(needle) + 1, step):
            window = haystack[start:start + len(needle)]
            best = max(best, SequenceMatcher(None, needle, window).ratio() * 100)
            if best >= 99.9:
                break
        return best


def reference_rank(reference: str | None, evidence: list[dict]) -> tuple[int | None, str | None, float]:
    """(1-based rank of the first chunk that overlaps the gold reference, match type, best score).

    The dataset's ``reference_string`` is usually a whole booklet section (often longer
    than a page chunk), so containment is checked in both directions: the shorter of
    (chunk, reference) must appear in the longer one, exactly or fuzzily.
    """
    if not reference or not reference.strip():
        return None, None, 0.0
    ref = normalize(reference)
    best_score = 0.0
    for rank, item in enumerate(evidence, start=1):
        chunk = normalize(item["text"])
        if len(chunk) < MIN_CHUNK_CHARS:
            continue
        needle, haystack = (chunk, ref) if len(chunk) <= len(ref) else (ref, chunk)
        if needle in haystack:
            return rank, "exact", 100.0
        score = _partial_ratio(needle, haystack)
        best_score = max(best_score, score)
        if score >= FUZZY_THRESHOLD:
            return rank, "fuzzy", score
    return None, None, best_score


# --------------------------------------------------------------------------- caching

def prediction_cache_key(claim: str, evidence: list[dict], config: ApiConfig) -> str:
    material = {
        "claim": claim,
        "evidence": [(e["chunk_id"], e["text"]) for e in evidence],
        "prompt_version": PROMPT_VERSION,
        "model": config.model,
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def classify_cached(claim: str, evidence: list[dict], config: ApiConfig, cache_dir: Path) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{prediction_cache_key(claim, evidence, config)}.json"
    if path.exists():
        return {**json.loads(path.read_text(encoding="utf-8")), "from_cache": True}
    result = classify_claim(claim, evidence, config)
    path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return {**result, "from_cache": False}


# --------------------------------------------------------------------------- encoder reuse

def install_encoder_cache() -> None:
    """retrieve_evidence() builds a SentenceTransformer per call; reuse one per config instead."""
    import sentence_transformers

    original = sentence_transformers.SentenceTransformer
    instances: dict[tuple, object] = {}

    def cached(model_name, *args, revision=None, device=None, **kwargs):
        key = (model_name, revision, device)
        if key not in instances:
            instances[key] = original(model_name, *args, revision=revision, device=device, **kwargs)
        return instances[key]

    sentence_transformers.SentenceTransformer = cached


# --------------------------------------------------------------------------- metrics

def prf(gold: list[str], pred: list[str]) -> dict:
    out = {}
    for label in LABELS:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    out["macro_f1"] = sum(out[l]["f1"] for l in LABELS) / len(LABELS)
    return out


def compute_metrics(records: list[dict], top_k: int) -> dict:
    with_ref = [r for r in records if r["has_reference"] and r["retrieval_status"] == "ok"]
    hits = [r for r in with_ref if r["reference_rank"] is not None]
    retrieval = {
        "top_k": top_k,
        "rows_with_reference": len(with_ref),
        f"recall@{top_k}": len(hits) / len(with_ref) if with_ref else None,
        "mrr": (sum(1 / r["reference_rank"] for r in hits) / len(with_ref)) if with_ref else None,
        "match_types": dict(Counter(r["reference_match_type"] for r in hits)),
        "by_language": {},
    }
    for lang in sorted({r["language"] for r in with_ref}):
        rows = [r for r in with_ref if r["language"] == lang]
        retrieval["by_language"][lang] = {
            "n": len(rows),
            f"recall@{top_k}": sum(1 for r in rows if r["reference_rank"] is not None) / len(rows),
        }
    retrieval["by_label"] = {}
    for label in LABELS:
        rows = [r for r in with_ref if r["gold_label"] == label]
        if rows:
            retrieval["by_label"][label] = {
                "n": len(rows),
                f"recall@{top_k}": sum(1 for r in rows if r["reference_rank"] is not None) / len(rows),
            }
    retrieval["by_cross_lingual"] = {}
    for flag, name in ((False, "same_language"), (True, "cross_lingual")):
        rows = [r for r in with_ref if r.get("cross_lingual") is flag]
        if rows:
            retrieval["by_cross_lingual"][name] = {
                "n": len(rows),
                f"recall@{top_k}": sum(1 for r in rows if r["reference_rank"] is not None) / len(rows),
            }
    retrieval["empty_candidate_sets"] = sum(1 for r in records if r.get("n_candidates") == 0)

    classified = [r for r in records if r.get("predicted_label")]
    errors = [r for r in records if r.get("classification_status") == "error"]
    classification = None
    if classified:
        gold = [r["gold_label"] for r in classified]
        pred = [r["predicted_label"] for r in classified]
        majority = Counter(gold).most_common(1)[0]
        classification = {
            "n_classified": len(classified),
            "n_errors": len(errors),
            "accuracy": sum(g == p for g, p in zip(gold, pred)) / len(gold),
            "accuracy_errors_as_wrong": sum(g == p for g, p in zip(gold, pred)) / (len(gold) + len(errors)),
            "majority_baseline": {"label": majority[0], "accuracy": majority[1] / len(gold)},
            **prf(gold, pred),
            "confusion": {g: dict(Counter(p for gg, p in zip(gold, pred) if gg == g)) for g in LABELS},
            "by_language": {},
            "from_cache": sum(1 for r in classified if r.get("from_cache")),
            "tokens": {
                "prompt": sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in classified),
                "completion": sum((r.get("usage") or {}).get("completion_tokens", 0) for r in classified),
            },
            "error_messages": dict(Counter(r["error"].split(":")[0] for r in errors)),
        }
        for lang in sorted({r["language"] for r in classified}):
            rows = [r for r in classified if r["language"] == lang]
            g = [r["gold_label"] for r in rows]
            p = [r["predicted_label"] for r in rows]
            classification["by_language"][lang] = {
                "n": len(rows), "accuracy": sum(a == b for a, b in zip(g, p)) / len(rows),
                "macro_f1": prf(g, p)["macro_f1"],
            }
        classification["by_cross_lingual"] = {}
        for flag, name in ((False, "same_language"), (True, "cross_lingual")):
            rows = [r for r in classified if r.get("cross_lingual") is flag]
            if rows:
                g = [r["gold_label"] for r in rows]
                p = [r["predicted_label"] for r in rows]
                classification["by_cross_lingual"][name] = {
                    "n": len(rows), "accuracy": sum(a == b for a, b in zip(g, p)) / len(rows),
                    "macro_f1": prf(g, p)["macro_f1"],
                }
    return {"n_rows": len(records),
            "gold_distribution": dict(Counter(r["gold_label"] for r in records)),
            "retrieval": retrieval, "classification": classification}


def metrics_markdown(m: dict, run_name: str) -> str:
    k = m["retrieval"]["top_k"]
    lines = [f"# Evaluation – {run_name}", "",
             f"Rows: {m['n_rows']} · Gold labels: {m['gold_distribution']}", "",
             "## Retrieval", "",
             "| Setup | n (with reference) | Recall@%d | MRR |" % k, "|---|---|---|---|"]
    r = m["retrieval"]
    rec = r[f"recall@{k}"]
    lines.append(f"| all | {r['rows_with_reference']} | {rec:.3f} | {r['mrr']:.3f} |" if rec is not None
                 else "| all | 0 | – | – |")
    for lang, v in r["by_language"].items():
        lines.append(f"| {lang} | {v['n']} | {v[f'recall@{k}']:.3f} | |")
    for name, v in r["by_cross_lingual"].items():
        lines.append(f"| {name} | {v['n']} | {v[f'recall@{k}']:.3f} | |")
    for label, v in r["by_label"].items():
        lines.append(f"| gold={label} | {v['n']} | {v[f'recall@{k}']:.3f} | |")
    lines.append(f"\nClaims with an empty BM25 candidate set (hybrid only): {r['empty_candidate_sets']}")
    c = m["classification"]
    if c:
        lines += ["", "## Classification", "",
                  "| Setup | n | Accuracy | Macro-F1 |", "|---|---|---|---|",
                  f"| majority baseline ({c['majority_baseline']['label']}) | {c['n_classified']} | "
                  f"{c['majority_baseline']['accuracy']:.3f} | – |",
                  f"| Apertus zero-shot | {c['n_classified']} | {c['accuracy']:.3f} | {c['macro_f1']:.3f} |"]
        for lang, v in c["by_language"].items():
            lines.append(f"| {lang} | {v['n']} | {v['accuracy']:.3f} | {v['macro_f1']:.3f} |")
        for name, v in c["by_cross_lingual"].items():
            lines.append(f"| {name} | {v['n']} | {v['accuracy']:.3f} | {v['macro_f1']:.3f} |")
        lines += ["", "Per class:", "", "| Label | P | R | F1 | support |", "|---|---|---|---|---|"]
        for label in LABELS:
            v = c[label]
            lines.append(f"| {label} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |")
        lines += ["", "Confusion (rows = gold, cols = predicted):", "",
                  "| gold \\ pred | " + " | ".join(LABELS) + " |", "|---|" + "---|" * len(LABELS)]
        for g in LABELS:
            lines.append(f"| {g} | " + " | ".join(str(c["confusion"][g].get(p, 0)) for p in LABELS) + " |")
        lines += ["", f"API errors: {c['n_errors']} ({c['error_messages']}) · "
                      f"from cache: {c['from_cache']} · tokens: {c['tokens']}"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- main

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export", action="store_true", help="download the dataset split to --input and exit")
    p.add_argument("--input", type=Path, default=Path("data/evaluation/train.jsonl"))
    p.add_argument("--split", default="train")
    p.add_argument("--run-name", default=time.strftime("run-%Y%m%d-%H%M%S"))
    p.add_argument("--output-dir", type=Path, default=Path("data/evaluation/runs"))
    p.add_argument("--cache-dir", type=Path, default=Path("data/evaluation/cache"))
    p.add_argument("--limit", type=int, help="stratified sample size (per language x label round-robin)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--retrieval-only", action="store_true", help="skip Apertus; Recall@k only")
    p.add_argument("--retrieval", choices=["hybrid", "dense"], default="hybrid")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--candidate-k", type=int, default=50)
    p.add_argument("--db", type=Path, default=Path("data/qdrant"))
    p.add_argument("--collection", default="voting_booklets_e5_v1")
    p.add_argument("--sparse-dir", type=Path, default=Path("data/bm25"))
    p.add_argument("--manifest", type=Path, default=Path("data/booklets/manifest.json"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--claim-field", default="claim")
    p.add_argument("--label-field", default="entailment_label")
    p.add_argument("--reference-field", default="reference_string")
    p.add_argument("--language-field", default="claim_language")
    p.add_argument("--reference-language-field", default="reference_language")
    p.add_argument("--url-field", default="booklet_url")
    p.add_argument("--label-map", type=Path, help="JSON file mapping raw label values to ENTAIL/NEUTRAL/CONTRADICT")
    p.add_argument("--sleep", type=float, default=0.0, help="seconds between API calls")
    p.add_argument("--resume", action="store_true",
                   help="continue an interrupted run: keep existing rows in predictions.jsonl, process the rest")
    args = p.parse_args()

    if args.export:
        export_dataset(args.input, split=args.split)
        return 0

    label_map = DEFAULT_LABEL_MAP if not args.label_map else json.loads(args.label_map.read_text())
    rows = load_rows(args.input)
    rows = [r for r in rows if r.get("split", args.split) == args.split]
    for r in rows:
        r["_gold"] = gold_label(r, label_map, args.label_field)
        r["_lang"] = str(r.get(args.language_field) or "?")
    print(f"{len(rows)} rows · labels {dict(Counter(r['_gold'] for r in rows))} · "
          f"languages {dict(Counter(r['_lang'] for r in rows))}")
    cross = sum(1 for r in rows if r.get(args.reference_language_field) not in (None, r["_lang"]))
    print(f"cross-lingual rows (claim language != reference language): {cross}/{len(rows)}")
    if args.limit:
        rows = stratified_sample(rows, args.limit, args.seed, key=lambda r: (r["_lang"], r["_gold"]))
        print(f"sampled {len(rows)} rows (seed {args.seed})")

    doc_ids = url_to_document_id(args.manifest)
    config = ApiConfig.from_env()
    install_encoder_cache()
    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"
    records: list[dict] = []
    done: set = set()
    if args.resume and predictions_path.exists():
        records = load_rows(predictions_path)
        done = {r["row_index"] for r in records}
        print(f"resuming: {len(done)} rows already in {predictions_path}")
    elif predictions_path.exists() and not args.resume:
        raise SystemExit(f"{predictions_path} exists; pass --resume to continue or choose another --run-name")

    with predictions_path.open("a" if args.resume else "w", encoding="utf-8") as out:
        for i, row in enumerate(rows, start=1):
            if row.get("row_index") in done:
                continue
            claim = str(row[args.claim_field]).strip()
            reference = row.get(args.reference_field)
            record = {
                "row_index": row.get("row_index"), "claim": claim, "language": row["_lang"],
                "gold_label": row["_gold"], "has_reference": bool(reference and str(reference).strip()),
                "reference_string": reference, "booklet_url": row.get(args.url_field),
                "reference_language": row.get(args.reference_language_field),
                "cross_lingual": (row.get(args.reference_language_field) is not None
                                  and row.get(args.reference_language_field) != row["_lang"]),
                "document_id": doc_ids.get(str(row.get(args.url_field, "")).strip()),
                "retrieval_status": "ok", "classification_status": "skipped",
            }
            try:
                if not record["document_id"]:
                    raise ValueError("booklet_url not in manifest")
                evidence, _, diagnostics = retrieve_evidence(
                    args.db, args.collection, record["document_id"], claim, args.top_k, args.device,
                    retrieval=args.retrieval, candidate_k=args.candidate_k, sparse_dir=args.sparse_dir)
                metadata = load_document_metadata(args.manifest, record["document_id"])
                evidence = attach_document_metadata(evidence, metadata)
                rank, match_type, score = reference_rank(reference, evidence)
                record.update({
                    "retrieved": [{"chunk_id": e["chunk_id"], "page": e["page_number"],
                                   "score": e.get("score"), "bm25_rank": e.get("bm25_rank")} for e in evidence],
                    "n_candidates": diagnostics.get("candidate_count"),
                    "reference_rank": rank, "reference_match_type": match_type,
                    "reference_best_score": round(score, 1),
                })
            except Exception as error:  # retrieval failure: record, do not classify
                record.update({"retrieval_status": "error", "error": str(error),
                               "reference_rank": None, "reference_match_type": None})
                evidence = []

            if not args.retrieval_only and record["retrieval_status"] == "ok":
                try:
                    result = classify_cached(claim, evidence, config, args.cache_dir)
                    record.update({
                        "classification_status": result["status"],
                        "predicted_label": result["label"],
                        "explanation": result.get("explanation"),
                        "citations": [{"evidence_id": c["evidence_id"], "page": c.get("page_number"),
                                       "quote": c["quote"]} for c in result.get("citations", [])],
                        "model": result.get("model"), "usage": result.get("usage"),
                        "from_cache": result.get("from_cache", False),
                    })
                    if not result.get("from_cache") and args.sleep:
                        time.sleep(args.sleep)
                except InferenceError as error:
                    record.update({"classification_status": "error", "error": str(error),
                                   "raw_model_output": error.details.get("raw_model_output")})
                except Exception as error:
                    record.update({"classification_status": "error", "error": f"{type(error).__name__}: {error}"})

            records.append(record)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            hit = "hit" if record.get("reference_rank") else ("-" if not record["has_reference"] else "MISS")
            pred = record.get("predicted_label") or ("–" if args.retrieval_only else record["classification_status"])
            print(f"[{i}/{len(rows)}] {row['_lang']} gold={record['gold_label']:<10} pred={pred:<10} "
                  f"ref={hit:<4} {claim[:70]}")

    metrics = {
        "run_name": args.run_name, "dataset": DATASET, "split": args.split,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "settings": {"retrieval": args.retrieval, "top_k": args.top_k, "candidate_k": args.candidate_k,
                     "collection": args.collection, "model": config.model, "prompt_version": PROMPT_VERSION,
                     "fuzzy_threshold": FUZZY_THRESHOLD, "limit": args.limit, "seed": args.seed,
                     "retrieval_only": args.retrieval_only},
        **compute_metrics(records, args.top_k),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "metrics.md").write_text(metrics_markdown(metrics, args.run_name), encoding="utf-8")
    print("\n" + metrics_markdown(metrics, args.run_name))
    print(f"-> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
