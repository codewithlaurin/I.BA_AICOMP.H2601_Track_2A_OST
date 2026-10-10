# Results

Scored with the organisers' `evaluate.py` (macro-F1, invalid lines count as wrong).

## Task B, full public dataset (1,495 rows, `OSTswiss/MNLIoverSwissVotingBooklets` train split)

Branch `feature/cli`, `python -m claimcheck`, model `swiss-ai/Apertus-v1.5-8B`, one call per case.
Scored with the organisers' `evaluate.py`. Network outages during the runs were repaired by
re-running the affected ids and merging (`output/cli-b-all*/predictions*.jsonl`).

| Run (2026-10-10) | Macro-F1 | E / N / C F1 | Notes |
| --- | --- | --- | --- |
| prompt v4, quote-grounding rule on (c21d7f0) | 0.887 | 0.913 / 0.887 / 0.860 | rule forced 80 cases to neutral; all 80 were gold E/C |
| prompt v4, rule removed (e54faee) | **0.938** | 0.949 / 0.958 / 0.908 | current code |
| prompt v5 "extra detail is not a contradiction" | 0.926 | 0.942 / 0.948 / 0.888 | contradiction recall 0.90 -> 0.87; reverted |

Current code: accuracy 0.938; per case 3,182 input / 136 output tokens, 1.0 s.

Macro-F1 by language pair (source -> claim), current code:

| | de claim | fr claim | it claim |
| --- | --- | --- | --- |
| de source | 0.90 | 0.93 | 0.92 |
| fr source | 0.97 | 0.97 | 0.94 |
| it source | 0.87 | 0.95 | 0.97 |

Caveat: 100 of these rows (`b-100`) were used to tune the prompt; the disjoint 100-row holdout
gives 0.92 with the current code.

## Task A, 30 rows, full-document baseline (no retrieval yet)

Macro-F1 0.79 (minimum 0.60), evidence score 0.32, ~31k input tokens per case.
Offline page-retrieval study: `multilingual-e5-small` with claim + vote finds the gold page
in the top 8 for 91 % of 429 rows (BM25: 76 %). Next step: send only the top pages.
