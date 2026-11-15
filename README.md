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

## Shared artifacts (Google Drive)

Large generated files are not in Git. They are shared in a private team folder:
<!-- TODO: add the Drive folder link -->
**Drive folder:** `<DRIVE_FOLDER_LINK>`

The folder holds one archive, `attvr_artifacts.zip`, with:

| Folder | Contents |
|---|---|
| `data/annotations/` | Splits, hard negatives, action contrast, opposite label pairs |
| `outputs/features/vjepa/` | V-JEPA 2 features, one `.npz` per video (193,690 files) |
| `outputs/features/clip/` | CLIP features for the test videos |
| `outputs/checkpoints/` | Best checkpoint of every trained model and seed |
| `outputs/results/` | Metrics JSON and per-query rankings |
| `logs/` | Slurm logs and per-epoch training logs |

Not on Drive:

- `data/videos/` and `data/raw_metadata/`: SSv2 terms do not allow
  redistribution. Download them from the official source (see
  [Prepare data](#prepare-data)). Videos are only needed to extract features
  again or to run the CLIP baseline on new videos.
- `outputs/model_cache/`: Hugging Face weights, downloaded on first use.

Download `attvr_artifacts.zip` into `artifacts/` (ignored by Git) and unpack from the repository
root. Paths in the zip are relative to the root, so files land in the folders
above:

```bash
unzip artifacts/attvr_artifacts.zip
```

To refresh the Drive copy after new runs, rebuild the zip and upload it again:

```bash
bash scripts/pack_artifacts.sbatch   # or: sbatch scripts/pack_artifacts.sbatch
```

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

**Note:** The `data/` directory is excluded from Git. Generate the annotation files locally from authorized official metadata, or unpack `annotations.tar.gz` from the [shared Drive folder](#shared-artifacts-google-drive).

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
without transitions (B2), and `all_ablations` runs `no_transition`,
`mean_transition`, `final_minus_initial`, and
`full_transition_without_query_conditioning` (B3, the main model's `[z, Δz]`
input without the reranker):

```bash
python -m src.models.baselines.vjepa_baseline --variant pooled all_ablations --seeds 42 43 44
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
| `bash scripts/submit_baselines.sh` | CLIP, every V-JEPA variant, then the comparison table |
| `sbatch scripts/extract_vjepa_features.sbatch` | Sharded V-JEPA extraction (array job) |
| `sbatch scripts/train_eval_single_stage.sbatch` | Train and evaluate one seed per array task |
| `sbatch scripts/train_single_stage.sbatch` / `scripts/eval_single_stage.sbatch` | Train or evaluate only; extra arguments are passed through |
| `sbatch scripts/eval_clip_baseline.sbatch` | CLIP baseline (B1) |
| `bash scripts/submit_main_experiment.sh` | E1 stage 1 and 2 (3 seeds), E2 evaluation, reverse-time for E1 and E2 |
| `sbatch scripts/train_token_matcher.sbatch` | Stage 2 token-level matcher; extra arguments are passed through |
| `sbatch scripts/train_vjepa_variants.sbatch` | Array: `pooled` (B2) and the ablations; `--array=0` runs B2 only |
| `sbatch scripts/build_tables.sbatch` | Comparison table and error-analysis figures |
| `sbatch scripts/pack_artifacts.sbatch` | `artifacts/attvr_artifacts.zip` for the [shared Drive folder](#shared-artifacts-google-drive) |

Run tests with `python -m pytest`. See [Tickets](docs/Tickets.md) for interfaces,
[PRD](docs/PRD.md) for project details, and [Owners](docs/OWNERS.md) for team responsibilities.
