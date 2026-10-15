"""Score predictions the way the benchmark evaluation does, as far as it can be reproduced offline.

Per task (A = document, B = reference):
- macro-F1 from the predicted labels only. Missing or invalid responses count as wrong.
- Task A evidence score: the share of gold entailment/contradiction cases where one of the
  first five evidence items, each at most about a page long, overlaps the gold reference
  passage (fuzzy match).
Token and time scores are relative to the other teams, so they cannot be computed locally;
the self-reported averages are shown for reference.

Verbatim copy of the organisers' evaluate.py (OST, 2026-10-09). Do not edit; use
scripts/evaluate.py for our own extended report.
"""

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from rapidfuzz import fuzz
from sklearn.metrics import f1_score, precision_recall_fscore_support

LABELS = ["entailment", "neutral", "contradiction"]
THRESHOLDS = {"A": 0.60, "B": 0.70}
INVALID = -1
MAX_QUOTES = 5
MAX_QUOTE_CHARS = 5000  # about one booklet page: 95% of pages are shorter


def read_jsonl(path):
    with path.open(encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]


def normalize(text):
    text = unicodedata.normalize("NFKC", text).replace("­", "")
    text = re.sub(r"(\w) ?-\s*\n\s*(\w)", r"\1\2", text)  # words hyphenated across lines
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_matches(quote, passage, min_ratio, max_chars):
    """A quote matches if it lies (almost) inside the gold passage or contains it.

    partial_ratio aligns the shorter text inside the longer one, so a short quote from the
    passage and a page that contains the passage both match. Longer quotes do not count.
    """
    quote, passage = normalize(quote), normalize(passage)
    if not quote or len(quote) > max_chars:
        return False
    return fuzz.partial_ratio(quote, passage) >= min_ratio


def task_of(case_id, case):
    if case:
        return "A" if "booklet" in case else "B"
    return case_id.rsplit("-", 1)[-1]


def label_of(prediction):
    """Return (label, problem). The label is INVALID when the response breaks the contract."""
    label = prediction.get("label")
    if not isinstance(label, int) or isinstance(label, bool) or label not in range(3):
        return INVALID, "invalid label"
    if prediction.get("label_name") != LABELS[label]:
        return INVALID, "label_name does not match label"
    return label, None


def quotes_of(prediction):
    evidence = prediction.get("evidence")
    if not isinstance(evidence, list):
        return []
    return [item["text"] for item in evidence[:MAX_QUOTES]
            if isinstance(item, dict) and isinstance(item.get("text"), str)]


def macro_f1(pairs):
    """pairs: (gold, predicted); predicted INVALID counts as wrong for every class."""
    if not pairs:
        return 0.0, {}
    gold, predicted = zip(*pairs)
    # Restricting to labels 0-2 makes INVALID a miss for the gold class, never a false positive.
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, predicted, labels=[0, 1, 2], zero_division=0
    )
    per_class = {
        LABELS[c]: {"precision": float(precision[c]), "recall": float(recall[c]), "f1": float(f1[c]), "support": int(support[c])}
        for c in range(3)
    }
    # Average over classes that occur in the gold labels or the predictions, so predicting a
    # class that is absent from a small sample still costs points.
    labels = sorted(set(gold) | {p for p in predicted if p != INVALID})
    macro = f1_score(gold, predicted, labels=labels, average="macro", zero_division=0)
    return float(macro), per_class


def main():
    parser = argparse.ArgumentParser(description="Score predictions: macro-F1 per task and the task A evidence score.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True, help="expected-labels.jsonl from prepare_cases.py")
    parser.add_argument("--cases", type=Path, help="cases.jsonl, to report scores per language pair")
    parser.add_argument("--min-ratio", type=float, default=90, help="fuzzy-match score (0-100) a quote needs")
    parser.add_argument("--max-quote-chars", type=int, default=MAX_QUOTE_CHARS,
                        help="longest evidence text that counts (about one page)")
    parser.add_argument("--json", type=Path, help="also write the full report as JSON")
    args = parser.parse_args()

    expected = {row["id"]: row for row in read_jsonl(args.expected)}
    cases = {row["id"]: row for row in read_jsonl(args.cases)} if args.cases else {}

    predictions, issues = {}, Counter()
    for prediction in read_jsonl(args.predictions):
        case_id = prediction.get("id")
        if case_id not in expected:
            issues["unknown id"] += 1
        elif case_id in predictions:
            issues["duplicate id"] += 1
            predictions[case_id] = None
        else:
            predictions[case_id] = prediction

    pairs = defaultdict(list)
    evidence = defaultdict(lambda: {"cases": 0, "found": 0, "missing_gold": 0})
    by_language = defaultdict(list)
    usage = Counter()
    for case_id, gold in expected.items():
        case = cases.get(case_id)
        task = task_of(case_id, case)
        prediction = predictions.get(case_id)
        predicted = INVALID
        if prediction is None:
            issues["missing or duplicate response"] += 1
        else:
            predicted, problem = label_of(prediction)
            if problem:
                issues[problem] += 1
            metrics = prediction.get("metrics")
            for key in ("input_tokens", "output_tokens", "inference_time_ms"):
                value = metrics.get(key) if isinstance(metrics, dict) else None
                if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
                    usage[key] += value
                else:
                    issues[f"invalid metrics.{key} (does not affect the score)"] += 1
        pairs[task].append((gold["label"], predicted))

        if task == "A" and gold["label"] != 1:
            stats = evidence[task]
            stats["cases"] += 1
            passage = gold.get("reference")
            if passage is None:
                stats["missing_gold"] += 1
            elif prediction and any(quote_matches(q, passage, args.min_ratio, args.max_quote_chars)
                                    for q in quotes_of(prediction)):
                stats["found"] += 1

        if case:
            source = case.get("reference", case.get("booklet", {})).get("language", "?")
            by_language[f"{task} {source}->{case['claim'].get('language', '?')}"].append((gold["label"], predicted))

    report = {"cases": len(expected), "responses": len(predictions), "issues": dict(issues), "tasks": {}}
    for task, task_pairs in sorted(pairs.items()):
        f1, per_class = macro_f1(task_pairs)
        result = {
            "cases": len(task_pairs),
            "macro_f1": f1,
            "threshold": THRESHOLDS.get(task),
            "passes_threshold": f1 >= THRESHOLDS[task] if task in THRESHOLDS else None,
            "accuracy": sum(g == p for g, p in task_pairs) / len(task_pairs),
            "per_class": per_class,
        }
        stats = evidence.get(task)
        if stats and stats["cases"]:
            scored = stats["cases"] - stats["missing_gold"]
            result["evidence_score"] = stats["found"] / stats["cases"] if not stats["missing_gold"] else None
            result["evidence"] = {"cases": stats["cases"], "found": stats["found"], "without_gold_passage": stats["missing_gold"]}
            if scored < stats["cases"]:
                result["evidence"]["note"] = "re-run prepare_cases.py to add gold passages to expected-labels.jsonl"
        report["tasks"][task] = result
    report["languages"] = {
        pair: {"cases": len(rows), "macro_f1": macro_f1(rows)[0]} for pair, rows in sorted(by_language.items())
    }
    report["usage_mean_per_case"] = {key: usage[key] / len(expected) for key in ("input_tokens", "output_tokens", "inference_time_ms")}

    names = {"A": "document", "B": "reference"}
    print(f"Cases: {len(expected)}  responses: {len(predictions)}")
    for task, result in report["tasks"].items():
        print(f"\nTask {task} ({names.get(task, '?')}, {result['cases']} cases)")
        line = f"  macro-F1:       {result['macro_f1']:.4f}"
        if result["threshold"] is not None:
            verdict = "pass" if result["passes_threshold"] else "BELOW"
            line += f"   (minimum {result['threshold']:.2f}: {verdict})"
        print(line)
        print(f"  accuracy:       {result['accuracy']:.4f}")
        for name, stats in result["per_class"].items():
            print(f"    {name:13} P={stats['precision']:.3f} R={stats['recall']:.3f} "
                  f"F1={stats['f1']:.3f} n={stats['support']}")
        if "evidence" in result:
            details = result["evidence"]
            if result["evidence_score"] is None:
                print(f"  evidence score: n/a, {details['without_gold_passage']} of {details['cases']} cases "
                      f"have no gold passage ({details['note']})")
            else:
                print(f"  evidence score: {result['evidence_score']:.4f}   "
                      f"({details['found']} of {details['cases']} entailment/contradiction cases cite the gold passage)")
    if report["languages"]:
        print("\nMacro-F1 by language (source->claim):")
        for pair, result in report["languages"].items():
            print(f"  {pair:10} {result['macro_f1']:.4f}  n={result['cases']}")
    print("\nSelf-reported usage per case: " + ", ".join(
        f"{key} {value:.0f}" for key, value in report["usage_mean_per_case"].items()))
    print("Token and time scores are relative to other teams and are computed only in the official run.")
    if issues:
        print("\nProblems found (invalid labels count as wrong):")
        for problem, count in issues.most_common():
            print(f"  {count:5}  {problem}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
