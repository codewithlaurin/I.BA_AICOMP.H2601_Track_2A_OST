# Results

Scored with the organisers' `evaluate.py` (macro-F1, invalid lines count as wrong).

## Task B, full public dataset (1,495 rows, `OSTswiss/MNLIoverSwissVotingBooklets` train split)

Run 2026-10-10, branch `feature/cli`, commit c21d7f0, `python -m claimcheck`, prompt `judge-nli-v4`,
model `swiss-ai/Apertus-v1.5-8B`, one call per case. Rows 432-1494 were re-run after a network
outage (1,063 cases), results merged (`output/cli-b-all/predictions.merged.jsonl`).

| Metric | Value |
| --- | --- |
| Macro-F1 | **0.887** (minimum 0.70) |
| Accuracy | 0.887 |
| Entailment P / R / F1 | 0.946 / 0.882 / 0.913 (n=498) |
| Neutral P / R / F1 | 0.822 / 0.964 / 0.887 (n=498) |
| Contradiction P / R / F1 | 0.911 / 0.816 / 0.860 (n=499) |
| Input / output tokens per case | 3,182 / 136 |
| Inference time per case | 1.0 s |

Macro-F1 by language pair (source -> claim):

| | de claim | fr claim | it claim |
| --- | --- | --- | --- |
| de source | 0.879 | 0.807 | 0.824 |
| fr source | 0.950 | 0.964 | 0.869 |
| it source | 0.831 | 0.862 | 0.957 |

Same-language pairs 0.88-0.96, cross-lingual 0.81-0.95. Weakest: German source with a
French or Italian claim. Main error: contradictions read as neutral (recall 0.82) and
neutral over-predicted (precision 0.82). Note: the prompt was tuned on 100 of these rows
(`b-100`), so this number is slightly optimistic; the disjoint 100-row holdout gave 0.90.

## Task A, 30 rows, full-document baseline (no retrieval yet)

Macro-F1 0.79 (minimum 0.60), evidence score 0.32, ~31k input tokens per case.
Offline page-retrieval study: `multilingual-e5-small` with claim + vote finds the gold page
in the top 8 for 91 % of 429 rows (BM25: 76 %). Next step: send only the top pages.
