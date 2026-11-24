# Accumulated Temporal Transitions for Text-Video Retrieval

This project explores text-to-video retrieval on Something-Something V2 using
ordered video frames and their temporal transitions.

**Research question.** Can accumulated latent transitions between consecutive
video states improve text-video retrieval compared with static video
representations?

**Final method.** V-JEPA 2 states `z[1:T]` → transitions `Δz[t] = z[t+1] - z[t]`
→ temporal self-attention over tokens `[z[t], Δz[t]]` → query-conditioned
token-level cross-attention matching → ranking. It is trained in two stages:
the temporal encoder with the cosine objective, then, with the encoder frozen,
the token-level matcher. The main experiment compares this model (E2) with an
identical model whose tokens carry no `Δz` (E1, `[z[t], 0]`); see
[Main experiment](#main-experiment-e1-vs-e2).

## Install

Run from the repository root with Conda installed:

```bash
conda create -n attvr python=3.11 -y
conda activate attvr
python -m pip install -r requirements.txt
```

## Reproduce all generated files

Large generated files (annotations, features, checkpoints, results, logs) are
not in Git. Rebuild them with one script, after putting the SSv2 metadata in
`data/raw_metadata/` and the videos in `data/videos/` (see
[Prepare data](#prepare-data)):

```bash
conda activate attvr
bash scripts/reproduce_all.sh      # all steps
bash scripts/reproduce_all.sh 4    # resume from step 4
```

| Step | What it runs | Writes |
|---|---|---|
| 1 | Splits and candidate sets | `data/annotations/` |
| 2 | V-JEPA 2 feature extraction (GPU) | `outputs/features/vjepa/` |
| 3 | CLIP baseline (B1), V-JEPA baseline (B2), ablations, E2 stage 1 | `outputs/features/clip/`, `outputs/checkpoints/`, `outputs/results/` |
| 4 | Single-stage transition model, 3 seeds | same |
| 5 | E1 stage 1 (states-only encoder), 3 seeds | same |
| 6 | Stage 2 token-level matcher for E2 and E1, then reverse-time evaluation | same |

Per-epoch training logs go to `logs/`. Hugging Face weights are downloaded to
`outputs/model_cache/` on first use.

All randomness is seeded. The split uses `--seed 42`, with the validation
entries sorted by video ID before the seeded permutation; the hard-negative and
action-contrast sets use `RANDOM_SEED` from `config.py` (42, override with
`ATTVR_RANDOM_SEED`); every model is trained with seeds 42, 43, and 44. With
these defaults, the splits and candidate sets come out byte-identical. Trained
checkpoints are seeded too, but GPU kernels are not bit-exact, so metrics can
differ slightly in the last digits.

On Slurm, submit the same steps in order, waiting for each to finish:
`bash scripts/submit_single_stage.sh` (steps 2 and 4),
`bash scripts/submit_baselines.sh` (step 3),
`sbatch scripts/train_token_matcher.sbatch --encoder full_transition --seeds 42 43 44`
(E2 stage 2), then `bash scripts/submit_main_experiment.sh` (steps 5 and 6 for
E1, and E2 evaluation with reverse time). Step 1 runs on CPU in a few minutes.

## Prepare data

Put videos in `data/videos/` and annotations in `data/annotations/train.jsonl`.
Each line in the annotation file is a JSON object, for example:

```json
{"query_id":"q1","video_id":"123","video_path":"data/videos/123.webm","text_query":"putting something on something","target_video_id":"123","split":"train"}
```

`video_path` is relative to the repository root. The local `data/` directory
is ignored by Git.

## Reproduce the SSv2 Dataset Splits

We use the official Something-Something V2 (SSv2) training and validation metadata to create reproducible train, validation, and test splits.

The official training set is preserved. The labeled official validation set is divided equally into validation and test subsets using random seed 42. The official hidden-label test set is not used.

### 1. Download the metadata

Obtain the official SSv2 metadata through an authorized source and place the following files in `data/raw_metadata/`:

* `train.json`
* `validation.json`

The original filenames, `something-something-v2-train.json` and `something-something-v2-validation.json`, are also supported.

### 2. Generate the annotations

Run the following command from the repository root:

```bash
python -m src.data.prepare_data --raw-metadata data/raw_metadata --seed 42
```

The script generates:

| File                           | Number of samples |
| ------------------------------ | ----------------: |
| `data/annotations/train.jsonl` |           168,913 |
| `data/annotations/val.jsonl`   |            12,388 |
| `data/annotations/test.jsonl`  |            12,389 |

These counts correspond to the official SSv2 metadata used for our project.

### 3. Load the annotations

The generated files follow the shared annotation schema and can be loaded using:

```python
from pathlib import Path
from src.data.dataset import load_annotations

train = load_annotations(Path("data/annotations/train.jsonl"))
val = load_annotations(Path("data/annotations/val.jsonl"))
test = load_annotations(Path("data/annotations/test.jsonl"))
```

The splitting procedure sorts the original validation entries by video ID before applying the seeded random permutation. This ensures that team members using the same official metadata and seed obtain identical splits.

The generated annotations reference video files under `data/videos/` by default. The corresponding SSv2 videos must be obtained separately to run video-based models.

**Note:** The `data/` directory is excluded from Git. Generate the annotation files locally from authorized official metadata.

### 4. Build the hard-negative and action-contrast sets

```bash
python -m src.data.build_candidate_sets
```

This reads `test.jsonl` and writes to `data/annotations/`:

- `hard_negatives.jsonl`: each query's target plus up to 9 negatives, chosen in
  order from opposite actions on the same object (`opposite_action`), other
  actions on a shared object (`same_objects`), and any other action
  (`other_action`).
- `action_contrast.jsonl`: negatives are all opposite-action videos. Queries
  whose action has no opposite are left out, and lists may be shorter than 10.
- `opposite_label_pairs.tsv`: the label pairs treated as opposites. They come
  from `OPPOSITE_LABEL_PAIRS` in `src/eval/evaluate.py`, which was checked by
  hand; review this file after changing that list.

It also writes `outputs/results/candidate_set_coverage.json`: the share of
negatives of each type and the number of queries with at least one
opposite-action negative. Existing candidate files are kept unless you pass
`--overwrite`, because results already computed on them would no longer match.


## Extract video features

```bash
python -m src.models.transition_model.pipelines.extract_features \
  --annotations data/annotations/train.jsonl data/annotations/val.jsonl data/annotations/test.jsonl \
  --batch-size 8 --num-workers 4
```

This writes one `.npz` per unique video to `outputs/features/vjepa/`. The first
run downloads the V-JEPA 2 model. Re-running skips videos that are already
cached, so an interrupted run can be resumed. Unreadable videos are skipped and
listed in `outputs/features/vjepa/extract_failures.jsonl`; add `--strict` to stop
at the first one instead.

To use several GPUs, start one process per GPU with the same `--num-shards` and
a different `--shard-index`:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.models.transition_model.pipelines.extract_features --num-shards 2 --shard-index 0 &
CUDA_VISIBLE_DEVICES=1 python -m src.models.transition_model.pipelines.extract_features --num-shards 2 --shard-index 1 &
```

For a quick check without downloading model weights:

```bash
python -m src.models.transition_model.pipelines.extract_features \
  --encoder mock --limit 2 --output-dir outputs/features/mock
```

## Train the single-stage transition model

This is the single-stage model (states + `Δz`, pooled query-conditioned
reranker). The final method is trained in two stages; see
[Main experiment](#main-experiment-e1-vs-e2).

```bash
for seed in 42 43 44; do
  python -m src.models.transition_model.pipelines.train --seed "$seed"
done
```

The model reads states and transitions together: token `t` is
`[z[t], z[t+1] - z[t]]`, with states and transitions normalized separately.
Training reads `train.jsonl` and `val.jsonl` and uses cached features in
`outputs/features/vjepa/`. It follows the shared protocol in `config.py`: at
most 50 epochs, stopping once validation MRR has not improved for 5 epochs, and
one run per seed (42, 43, 44). Each run keeps its best checkpoint at
`outputs/checkpoints/transition_model_seed<seed>_best.pt` and writes per-epoch
loss and validation metrics to
`logs/transition_model_seed<seed>_train_log.jsonl`. The
checkpoint's `training_summary` records the best epoch, the epochs run, and
whether the run hit the epoch limit while still improving.

Useful options (run with `--help` for all of them):

- `--run-name NAME` changes the checkpoint and log file prefix, so different
  runs do not overwrite each other. The seed is always appended.
- `--video-input transition` feeds transitions only (the transition-only
  ablation). Give it its own `--run-name`.
- `--no-query-conditioning` drops the reranker (the B3 ablation). Give it its
  own `--run-name`.
- `--epochs` sets the epoch limit, and `--patience 0` disables early stopping.
- `--selection-metric recall@1` selects the checkpoint by Recall@1 instead of MRR.
- `--model-dim`, `--num-heads`, `--num-layers`, `--dropout`, and
  `--rerank-hidden-dim` set the model size.
- `--text-model-name` swaps in another mean-pooled Hugging Face text encoder.
- `--limit 200` runs on a small subset.

On Slurm, `RUN_NAME=transition_model sbatch scripts/train_eval_single_stage.sbatch` trains and
evaluates all three seeds as an array job (see [Run on Slurm](#run-on-slurm)).

## Evaluate

```bash
python -m src.models.transition_model.pipelines.evaluate
```

This evaluates every seed of `--run-name` (default `transition_model`) on
`test.jsonl` and writes `outputs/results/transition_model_seed<seed>.json`.
Each result file holds:

- `full`: Recall@1/5/10, MRR, and mAP over all test videos.
- `hard_negative`: the same metrics on `data/annotations/hard_negatives.jsonl`,
  plus `by_negative_type` and `coverage`. For each negative type
  (`opposite_action`, `same_objects`, `other_action`), `by_negative_type` gives
  the pairwise accuracy (share of target-vs-negative pairs the target wins) and
  Recall@1/MRR against the target and only that type's negatives.
- `action_contrast`: the same metrics on
  `data/annotations/action_contrast.jsonl`, whose negatives are all opposite
  actions. It is `null` until that file is built (see
  [Build the hard-negative and action-contrast sets](#4-build-the-hard-negative-and-action-contrast-sets)).

The result file also records the seed, best epoch, epochs run, checkpoint path,
model and text-encoder settings, split, command, and timestamp. Use
`--checkpoint PATH --output PATH` to evaluate a single checkpoint instead, and
`--transition-dir outputs/features/transitions` to also save per-video
transition caches.

## Main experiment (E1 vs E2)

Both models use the same temporal encoder, token-level matcher, two-stage
training, protocol, seeds, and evaluation; only the `Δz` half of each token
differs. E1's tokens `[z[t], 0]` (`t = 1..T`) have the same width as E2's
`[z[t], Δz[t]]` (`t = 1..T-1`), so both models have the same number of
parameters, and E1 sees every state E2 can recover.

Stage 1 trains the encoder with the cosine objective only (no reranker); stage 2
freezes it and trains the token-level matcher. E2's stage 1 is the
`ablation_full_transition_without_query_conditioning` checkpoint. E1:

```bash
for seed in 42 43 44; do
  python -m src.models.transition_model.pipelines.train --seed "$seed" \
    --run-name ablation_state_only_without_query_conditioning \
    --video-input state_only --no-query-conditioning
done
python -m src.models.transition_model.pipelines.train_token_reranker --encoder state_only --seeds 42 43 44
```

Stage 2 loads `outputs/checkpoints/<stage-1 run>_seed<seed>_best.pt`, freezes
it, and writes `two_stage_token_reranker[_state_only]_seed<seed>.json`, plus
`per_query_<run>_seed<seed>.jsonl` (each test query's `query_id` and target rank
in the full, hard-negative, and action-contrast settings, for paired
comparisons) and `rankings_<run>_seed<seed>.jsonl`. `--eval-only` re-evaluates
trained checkpoints without retraining (an existing result file is kept and only
checked against the new metrics); `--reverse-time` evaluates them with each
video's states reversed in time (`<run>_seed<seed>_reversed.json`).

On Slurm, `bash scripts/submit_main_experiment.sh` submits E1 (both stages),
E2 evaluation, and reverse-time evaluation of both.

## Baselines, ablations, and error analysis

Every runner below scores `test.jsonl`, writes metrics to
`outputs/results/<model>.json` (`<model>_seed<seed>.json` for trained V-JEPA
variants), and writes per-query rankings to `outputs/results/rankings_<...>.jsonl`
for error analysis.

CLIP frame averaging (B1, zero-shot; caches features in `outputs/features/clip/`):

```bash
python -m src.models.baselines.clip_baseline
```

V-JEPA baseline and transition ablations. Each variant trains the shared
retrieval model on a different view of the cached V-JEPA states with the same
training protocol as the main model (epoch limit, early stopping, and selection
metric from `config.py`), then evaluates it on the full split, the
hard-negative set, and the action-contrast set. `pooled` is the V-JEPA baseline
without transitions (B2), and `all_ablations` runs the six T3.3 variants:

- `no_transition`: ordered states `z[1:T]`, the matched control for the main model.
- `transition_only`: transitions `Δz` without states.
- `shuffled_states`: `no_transition` with the state order permuted per video;
  redrawn on every training load (so every epoch), fixed per video id in evaluation.
- `mean_transition` and `final_minus_initial`: one summary token each.
- `full_transition_without_query_conditioning` (B3): the main model's `[z, Δz]`
  input without the reranker.

```bash
python -m src.models.baselines.vjepa_baseline --variant pooled all_ablations --seeds 42 43 44
```

Re-evaluate the trained checkpoints with each video's states reversed in time
before the variant input is built (no training; writes
`<model>_seed<seed>_reversed.json` and the matching rankings):

```bash
python -m src.models.baselines.vjepa_baseline --variant all_ablations --reverse-time --seeds 42 43 44
```

Checkpoints are saved as `outputs/checkpoints/<model>_seed<seed>_best.pt`. To
write rankings for the main model, evaluate its checkpoints with
`--variant full_transition` (it reads
`outputs/checkpoints/transition_model_seed<seed>_best.pt`, skips training, and
writes `rankings_transition_model_seed<seed>.jsonl` plus
`full_transition_seed<seed>.json`).
Pass `--checkpoint <path>` with one seed to evaluate any other saved variant
without retraining.

Categorize retrieval failures, compare models, and save example figures:

```bash
python -m src.models.baselines.error_analysis --figure-dir outputs/results/qualitative
```

## Run on Slurm

Job scripts live in `scripts/`. Submit them from the repository root: each one
sources `scripts/env.sh`, which activates the `attvr` Conda env (override with
`ATTVR_CONDA_ENV`). Slurm logs and per-epoch training logs go to `logs/`
(override the training-log folder with `ATTVR_LOG_ROOT` or `--log-dir`).

| Script | What it runs |
|---|---|
| `bash scripts/submit_single_stage.sh [--skip-extract]` | V-JEPA extraction, then train + evaluate for seeds 42, 43, 44 |
| `bash scripts/submit_baselines.sh` | CLIP, every V-JEPA variant, their reverse-time evaluation, then the comparison table |
| `sbatch scripts/extract_vjepa_features.sbatch` | Sharded V-JEPA extraction (array job) |
| `sbatch scripts/train_eval_single_stage.sbatch` | Train and evaluate one seed per array task |
| `sbatch scripts/train_single_stage.sbatch` / `scripts/eval_single_stage.sbatch` | Train or evaluate only; extra arguments are passed through |
| `sbatch scripts/eval_clip_baseline.sbatch` | CLIP baseline (B1) |
| `bash scripts/submit_main_experiment.sh` | E1 stage 1 and 2 (3 seeds), E2 evaluation, reverse-time for E1 and E2 |
| `sbatch scripts/train_token_matcher.sbatch` | Stage 2 token-level matcher; extra arguments are passed through |
| `sbatch scripts/train_vjepa_variants.sbatch` | Array: `pooled` (B2) and the six ablations, seeds 42-44; `--array=0` runs B2 only; `--reverse-time` re-evaluates |
| `sbatch scripts/build_tables.sbatch` | Comparison table and error-analysis figures |
| `bash scripts/reproduce_all.sh [step]` | Every step above in order on one machine (see [Reproduce all generated files](#reproduce-all-generated-files)) |

Run tests with `python -m pytest`. See [Tickets](docs/Tickets.md) for interfaces,
[PRD](docs/PRD.md) for project details, and [Owners](docs/OWNERS.md) for team responsibilities.
