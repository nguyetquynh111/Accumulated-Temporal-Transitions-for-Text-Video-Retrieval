# T4.2 — Reproducibility Checklist

**Project:** Accumulated Temporal Transitions for Text-Video Retrieval<br>
**Dataset:** Something-Something V2 (SSv2)<br>
**Evaluation split:** Test<br>
**Sources:** `data/annotations/`, `outputs/results/`, `src/eval/evaluate.py`

## 1. Dataset and evaluation-set checks

- [x] Training annotations: **168,913** records (`data/annotations/train.jsonl`).
- [x] Validation annotations: **12,388** records (`data/annotations/val.jsonl`).
- [x] Test annotations: **12,389** records (`data/annotations/test.jsonl`).
- [x] Total: **193,690** records across those three files.
- [x] **174** action labels in `outputs/results/dataset_stats.json`; the recorded per-split label totals match the respective split counts.
- [x] Hard-negative evaluation: **12,389** query records (`data/annotations/hard_negatives.jsonl`).
- [x] Action-contrast evaluation: **3,793** query records (`data/annotations/action_contrast.jsonl`).
- [x] Split sizes were checked against actual nonblank JSONL lines and the saved `dataset_stats.json` counts.

The hard-negative and action-contrast sets are **test evaluation settings**, not additional training samples. The saved dataset-statistics file reports zero missing videos; this is not an independent inspection of all raw video files in the current workspace.

## 2. Final results and table generation

- [x] **61** top-level model-evaluation JSON summaries were processed (excluding `dataset_stats.json`, `candidate_set_coverage.json`, `e1_e2_paired_bootstrap.json`, and `error_analysis.json`).
- [x] The generated `outputs/results/comparison_table.csv` contains **21** model-configuration rows, all labeled **test**.
- [x] Available seeds and run counts are included in the comparison table; the CLIP baseline has one result with no recorded seed, while most other configurations have seeds **42, 43, 44**.
- [x] Forward-time and reverse-time configurations appear as separate rows.
- [x] Missing metrics are marked **N/A**, rather than filled with estimated values.
- [x] Report-ready dataset, main-model, ablation, reverse-time, and statistical tables were prepared in `docs/t4_2_final_tables.md`.
- [x] `outputs/results/error_analysis.json` and qualitative figures are present as report inputs; their examples require a separate content review before inclusion in the final report.

**Do not commit `outputs/results/comparison_table.csv`**; T4.2 specifies that it is generated locally. Commit the reviewed Markdown report documents instead.

### Regenerate the comparison CSV

Run from the repository root in a Python environment containing the project's dependencies:

```python
from pathlib import Path
from src.eval.evaluate import build_comparison_table

results_dir = Path("outputs/results")
excluded = {
    "dataset_stats.json",
    "candidate_set_coverage.json",
    "e1_e2_paired_bootstrap.json",
    "error_analysis.json",
}
result_paths = sorted(
    path for path in results_dir.glob("*.json")
    if path.name not in excluded
)
build_comparison_table(
    result_paths=result_paths,
    output_path=results_dir / "comparison_table.csv",
)
```

The paired-bootstrap columns require the matching `per_query_two_stage_token_reranker_seed{42,43,44}.jsonl` and `per_query_two_stage_token_reranker_state_only_seed{42,43,44}.jsonl` files beside the summaries.

## 3. Matched E1/E2 verification

- [x] Compared **E1** (`two_stage_token_reranker_state_only`) against **E2** (`two_stage_token_reranker`) for matched seeds **42, 43, 44** on the **test** split.
- [x] Confirmed six corresponding per-query JSONL files are present in the latest merged repository.
- [x] Saved statistical report: `outputs/results/e1_e2_paired_bootstrap.json`.
- [x] **10,000** paired-query bootstrap resamples per seed, bootstrap random seed **42**.
- [x] Action-contrast Recall@1: E1 **72.12%**, E2 **75.86%** (three-seed means), difference **+3.73 percentage points**.
- [x] Action-contrast per-seed E2 − E1 95% confidence intervals match the generated comparison CSV:

| Test setting | Seed | E2 − E1 Recall@1 (pp) | Paired-bootstrap 95% CI (pp) | Paired queries |
|---|---:|---:|---|---:|
| Action-contrast | 42 | +4.0865 | [2.8474, 5.3520] | 3,793 |
| Action-contrast | 43 | +4.9038 | [3.6646, 6.1693] | 3,793 |
| Action-contrast | 44 | +2.2146 | [1.0276, 3.4537] | 3,793 |

- [x] The saved statistical report marks the **action-contrast** and **hard-negative** three-seed criteria as met.
- [x] The saved statistical report marks the **full-retrieval** three-seed criterion as **not met**; do not claim universal statistical improvement.

These results support improved **action-sensitive retrieval under the evaluated protocol**. Historical experiment-time candidate-file hashes were not independently verified, although both E1 and E2 evaluation code paths reference the same candidate-set path.

## 4. Remaining checks before final report integration

- [x] Ran the complete automated test suite using `python -m pytest -q` after incorporating commit `8cedbe8` from `origin/main`. All 60 tests and 4 subtests passed successfully.
- [ ] Review the complete `docs/t4_2_final_tables.md` Markdown preview for formatting and source/value accuracy.
- [ ] Inspect the contents of `outputs/results/error_analysis.json` and select any qualitative examples with Hao/Quynh.
- [ ] Check whether `full_transition_reversed` and the forward `transition_model` are directly comparable before reporting a forward-minus-reverse difference.
- [ ] If historical candidate-set snapshots are available, compare their checksums to the experiment-time versions.
- [ ] Obtain teammate review of final report integration.

**Scope:** These checks validate saved annotations and evaluation artifacts. T4.2 does not rerun feature extraction, retrain models, or independently reproduce all training jobs.
