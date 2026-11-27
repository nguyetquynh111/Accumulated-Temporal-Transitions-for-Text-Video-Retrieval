# T4.2 — Final Report Tables

## 1. Dataset Statistics

**Dataset:** Something-Something V2 (SSv2)

| Split | Samples | Action Classes |
|---|---:|---:|
| Training | 168,913 | 174 |
| Validation | 12,388 | 174 |
| Test | 12,389 | 174 |
| **Total** | **193,690** | **174** |

Counts were verified against the JSONL split files and `outputs/results/dataset_stats.json`.

## 2. Evaluation Query Statistics

| Evaluation Setting | Split | Queries |
|---|---|---:|
| Full retrieval | Test | 12,389 |
| Hard-negative | Test | 12,389 |
| Action-contrast | Test | 3,793 |

The evaluation sets are drawn from the test split. Their query counts do not represent additional training samples.

Query counts were verified against `data/annotations/test.jsonl`, `data/annotations/hard_negatives.jsonl`, and `data/annotations/action_contrast.jsonl`.


## 3. Main Model Performance

**Dataset:** Something-Something V2

**Split:** Test

Results are reported as mean Recall@1 across available seeds.
Where multiple runs are available, the standard deviation is
shown in percentage points.

| Model | Seeds | Full R@1 | Hard-Negative R@1 | Action-Contrast R@1 |
|---|---|---:|---:|---:|
| CLIP baseline | N/A | 4.01% | 33.51% | N/A |
| Pooled baseline | 42,43,44 | 9.29% ± 0.29 pp | 70.76% ± 0.41 pp | 61.09% ± 0.22 pp |
| Transition model | 42,43,44 | 9.87% ± 0.19 pp | 77.06% ± 0.14 pp | 73.93% ± 0.78 pp |
| E1: State-only reranker | 42,43,44 | 10.57% ± 0.57 pp | 76.96% ± 0.25 pp | 72.12% ± 1.17 pp |
| E2: States + transitions | 42,43,44 | 10.89% ± 0.16 pp | 78.06% ± 0.13 pp | 75.86% ± 1.13 pp |

**Evaluation query counts:** Full retrieval = 12,389; hard-negative = 12,389; action-contrast = 3,793.

N/A indicates that the corresponding evaluation metric
was not available in the saved results.

**Source:** `outputs/results/comparison_table.csv`,
generated from the saved experiment-result JSON files.


## 4. Ablation Study

**Dataset:** Something-Something V2

**Split:** Test

**Evaluation:** Forward-time retrieval

Results are mean Recall@1 across the available seeds.
Standard deviations are reported in percentage points.

| Ablation | Seeds | Full R@1 | Hard-Negative R@1 | Action-Contrast R@1 |
|---|---|---:|---:|---:|
| No transition | 42,43,44 | 9.72% ± 0.17 pp | 75.57% ± 0.24 pp | 70.74% ± 0.72 pp |
| Mean transition | 42,43,44 | 5.90% ± 0.24 pp | 66.87% ± 0.43 pp | 68.03% ± 0.76 pp |
| Final minus initial | 42,43,44 | 5.98% ± 0.22 pp | 66.63% ± 0.50 pp | 68.15% ± 1.05 pp |
| Shuffled states | 42,43,44 | 9.70% ± 0.09 pp | 74.31% ± 0.28 pp | 68.27% ± 0.42 pp |
| Transition only | 42,43,44 | 7.19% ± 0.29 pp | 74.11% ± 0.32 pp | 72.02% ± 0.45 pp |
| Full transition without query conditioning | 42,43,44 | 10.35% ± 0.31 pp | 77.06% ± 0.16 pp | 74.53% ± 1.15 pp |
| State-only without query conditioning | 42,43,44 | 10.03% ± 0.46 pp | 75.93% ± 0.65 pp | 70.94% ± 1.89 pp |

**Note:** The "No transition" ablation is different from the matched E1 state-only baseline. E1 is independently trained using the same model architecture as E2, with the temporal-transition channel set to zero. The main E1/E2 comparison is reported in Sections 3 and 6.

**Evaluation query counts:** Full retrieval = 12,389; hard-negative = 12,389; action-contrast = 3,793.

**Source:** `outputs/results/comparison_table.csv`, generated from saved ablation-result JSON files.


## 5. Reverse-Time Evaluation

**Dataset:** Something-Something V2

**Split:** Test

**Evaluation:** Reverse-time retrieval

Results are mean Recall@1 across seeds 42, 43, and 44.
Standard deviations are reported in percentage points.

| Model Configuration | Seeds | Full R@1 | Hard-Negative R@1 | Action-Contrast R@1 |
|---|---|---:|---:|---:|
| `final_minus_initial_reversed` | 42,43,44 | 0.00% ± 0.00 pp | 1.38% ± 0.07 pp | 2.04% ± 0.17 pp |
| `full_transition_reversed` | 42,43,44 | 2.56% ± 0.05 pp | 44.56% ± 0.52 pp | 23.12% ± 0.54 pp |
| `full_transition_without_query_conditioning_reversed` | 42,43,44 | 2.83% ± 0.14 pp | 45.14% ± 0.22 pp | 23.45% ± 0.37 pp |
| `mean_transition_reversed` | 42,43,44 | 0.00% ± 0.00 pp | 1.25% ± 0.10 pp | 1.76% ± 0.15 pp |
| `no_transition_reversed` | 42,43,44 | 5.91% ± 0.38 pp | 60.29% ± 1.47 pp | 38.95% ± 3.12 pp |
| `shuffled_states_reversed` | 42,43,44 | 9.59% ± 0.03 pp | 74.29% ± 0.32 pp | 68.24% ± 0.37 pp |
| `transition_only_reversed` | 42,43,44 | 0.56% ± 0.02 pp | 25.49% ± 0.43 pp | 10.70% ± 0.68 pp |
| `two_stage_token_reranker_reversed` | 42,43,44 | 2.95% ± 0.10 pp | 44.93% ± 0.35 pp | 23.67% ± 0.15 pp |
| `two_stage_token_reranker_state_only_reversed` | 42,43,44 | 5.45% ± 0.33 pp | 58.25% ± 1.27 pp | 33.52% ± 1.01 pp |

**Evaluation query counts:** Full retrieval = 12,389; hard-negative = 12,389; action-contrast = 3,793.

Model names are preserved from the saved evaluation results.
The `full_transition_reversed` configuration is listed separately;
its correspondence to the forward `transition_model` configuration
should be checked before calculating a forward-versus-reverse difference.

**Source:** `outputs/results/comparison_table.csv`,
generated from saved reverse-time experiment JSON files.


## 6. E1 vs. E2 Statistical Significance

**Dataset:** Something-Something V2

**Split:** Test

**E1:** State-only reranker

**E2:** States + temporal transitions reranker

**Seeds:** 42, 43, 44

**Bootstrap:** 10,000 paired-query resamples per seed

### 6.1 Overall Performance

| Evaluation Setting | Queries | E1 R@1 | E2 R@1 | E2 - E1 | Criterion Met |
|---|---:|---:|---:|---:|---|
| Full retrieval | 12,389 | 10.57% | 10.89% | +0.31 pp | No |
| Hard-negative | 12,389 | 76.96% | 78.06% | +1.10 pp | Yes |
| Action-contrast | 3,793 | 72.12% | 75.86% | +3.73 pp | Yes |

Values are means across the three seeds.
Differences are expressed in percentage points (pp).

### 6.2 Per-Seed Paired-Bootstrap Results

| Evaluation Setting | Seed | E2 - E1 (pp) | 95% CI (pp) | Queries |
|---|---:|---:|---|---:|
| Full retrieval | 42 | -0.0081 | [-0.5892, 0.5812] | 12,389 |
| Full retrieval | 43 | +0.9363 | [0.3794, 1.5013] | 12,389 |
| Full retrieval | 44 | +0.0081 | [-0.5569, 0.5489] | 12,389 |
| Hard-negative | 42 | +1.0816 | [0.3874, 1.7677] | 12,389 |
| Hard-negative | 43 | +1.2269 | [0.5489, 1.8888] | 12,389 |
| Hard-negative | 44 | +0.9767 | [0.3067, 1.6386] | 12,389 |
| Action-contrast | 42 | +4.0865 | [2.8474, 5.3520] | 3,793 |
| Action-contrast | 43 | +4.9038 | [3.6646, 6.1693] | 3,793 |
| Action-contrast | 44 | +2.2146 | [1.0276, 3.4537] | 3,793 |

### 6.3 Interpretation

The action-contrast results satisfy the project's primary
statistical criterion: E2 improves Recall@1 over E1 for
all three seeds, with each paired-bootstrap 95% confidence
interval excluding zero.

The hard-negative evaluation also satisfies this criterion.

The full-retrieval evaluation does not satisfy the
criterion across all three seeds. Therefore, the results
do not establish a consistent improvement for full retrieval.

The strongest supported conclusion is that temporal
transitions improve action-sensitive retrieval under
the evaluated conditions.

**Source:** `outputs/results/e1_e2_paired_bootstrap.json`
and `outputs/results/comparison_table.csv`.