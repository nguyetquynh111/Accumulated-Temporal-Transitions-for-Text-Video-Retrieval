# T3.2 — Model Comparison and Statistical Evaluation

## 1. Overview

This document summarizes the main E1 versus E2 comparison for the Accumulated Temporal Transitions for Text-Video Retrieval project.

The objective is to evaluate whether explicitly modeling temporal transitions improves action-sensitive text-video retrieval.

Two independently trained models are compared:

- **E1 — State-only baseline:** Uses video states without explicit temporal transitions.
- **E2 — Proposed model:** Uses video states together with accumulated temporal transitions.

Both models use the same two-stage architecture and evaluation pipeline. The intended difference is the inclusion of explicit temporal transitions (Δz) in E2.

## 2. Evaluation Setup

- **Dataset:** Something-Something V2 (SSv2)
- **Evaluation split:** Test
- **Total test queries:** 12,389
- **Action-contrast queries:** 3,793
- **Training seeds:** 42, 43, 44
- **Primary metric:** Action-contrast Recall@1
- **Statistical method:** Paired bootstrap
- **Bootstrap resamples:** 10,000 per seed
- **Bootstrap random seed:** 42
- **Confidence level:** 95%

The evaluation uses query-level retrieval ranks from the six E1/E2 result files.

Queries without an action-contrast rank are excluded. For each seed, E1 and E2 results are matched using `query_id`.

## 3. Main Results

The table below reports E2 minus E1 in action-contrast Recall@1.

| Seed | E2 − E1 (percentage points) | Paired-Bootstrap 95% CI | Paired Queries |
|---|---:|---|---:|
| 42 | +4.086 | [2.8474, 5.3520] | 3,793 |
| 43 | +4.904 | [3.6646, 6.1693] | 3,793 |
| 44 | +2.215 | [1.0276, 3.4537] | 3,793 |

**Mean action-contrast Recall@1 across three seeds:**

| Model | Mean Recall@1 |
|---|---:|
| E1 — State-only | 72.12% |
| E2 — Proposed transitions | 75.86% |
| E2 − E1 | +3.74 percentage points |

## 4. Statistical Interpretation

E2 achieves higher action-contrast Recall@1 than E1 for all three training seeds.

All three paired-bootstrap 95% confidence intervals exclude zero.

These results provide statistical evidence that explicitly incorporating temporal transitions improves action-contrast text-video retrieval compared with the matched state-only baseline.

The conclusion applies specifically to the action-contrast evaluation setting. Improvements in general full-split retrieval should be assessed separately.

## 5. Reproducibility

The comparison table is generated using:

`src/eval/evaluate.py`

Required inputs include:

- Saved experimental metrics JSON files.
- E1 per-query files for seeds 42, 43, and 44.
- E2 per-query files for seeds 42, 43, and 44.

The six per-query files follow these naming conventions:

- `per_query_two_stage_token_reranker_state_only_seed{seed}.jsonl`
- `per_query_two_stage_token_reranker_seed{seed}.jsonl`

All files should be placed in:

`outputs/results/`

The comparison table can then be regenerated using `build_comparison_table()`.

Output:

`outputs/results/comparison_table.csv`

The six per-query files and generated CSV are retained locally rather than committed to Git.

## 6. Validation and Limitations

- All six per-query files contain 12,389 test queries.
- Each file includes 3,793 action-contrast queries with non-null ranks.
- E1 and E2 query IDs are matched for the paired comparison.
- Calculated Recall@1 improvements agree with the saved experiment summaries.
- The repository uses the same action-contrast candidate-set path for both E1 and E2.
- The exact candidate IDs used in the historical evaluations have not been independently hash-verified.

The statistical conclusions assume that both models were evaluated using identical candidate sets, as intended by the shared evaluation pipeline.

## 7. Conclusion

The proposed E2 model consistently outperforms the matched E1 state-only baseline on action-contrast retrieval across three seeds.

All three paired-bootstrap confidence intervals exclude zero, supporting the project's hypothesis that explicit temporal-transition representations provide useful information for distinguishing action-sensitive videos.

These results will be used in the Milestone 4 final comparison tables and project report.