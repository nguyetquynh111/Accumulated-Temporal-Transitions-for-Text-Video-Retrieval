# Accumulated Temporal Transitions for Text-Video Retrieval

This project explores text-to-video retrieval on Something-Something V2 using
ordered video frames and their temporal transitions.

## Install

Run from the repository root with Conda installed:

```bash
conda create -n attvr python=3.11 -y
conda activate attvr
python -m pip install -r requirements.txt
```

## Prepare data

Put videos in `data/videos/` and annotations in `data/annotations/train.jsonl`.
Each line in the annotation file is a JSON object, for example:

```json
{"query_id":"q1","video_id":"123","video_path":"data/videos/123.webm","text_query":"putting something on something","target_video_id":"123","split":"train"}
```

`video_path` is relative to the repository root. The local `data/` directory
is ignored by Git.

## Extract video features

```bash
python -m src.models.transition_model.pipelines.extract_features
```

This reads `data/annotations/train.jsonl` and writes feature files to
`outputs/features/vjepa/`. The first run downloads the V-JEPA 2 model.

For a quick check without downloading model weights:

```bash
python -m src.models.transition_model.pipelines.extract_features \
  --encoder mock --limit 2 --output-dir outputs/features/mock
```

Extract features for every split you train or evaluate on (pass
`--annotations data/annotations/val.jsonl`, and likewise for `test.jsonl`).

## Train the transition model

```bash
python -m src.models.transition_model.pipelines.train
```

This reads `train.jsonl` and `val.jsonl`, trains on cached features in
`outputs/features/vjepa/`, and keeps the checkpoint with the best validation MRR
at `outputs/checkpoints/transition_model_best.pt`. Per-epoch loss and
validation metrics go to `outputs/checkpoints/train_log.jsonl`. Add
`--no-query-conditioning` to train the reranker ablation, and `--limit 200` for
a quick run on a small subset.

## Evaluate

```bash
python -m src.models.transition_model.pipelines.evaluate
```

This scores `test.jsonl` and writes `outputs/results/transition_model_full.json`
with Recall@1/5/10, MRR, and mAP. Hard-negative metrics are added when
`data/annotations/hard_negatives.jsonl` exists. Add
`--transition-dir outputs/features/transitions` to also save per-video
transition caches.

## Baselines, ablations, and error analysis

Every runner below scores `test.jsonl`, writes metrics to
`outputs/results/<model>.json`, and writes per-query rankings to
`outputs/results/rankings_<model>.jsonl` for error analysis.

CLIP frame averaging (B1, zero-shot; caches features in `outputs/features/clip/`):

```bash
python -m src.models.baselines.clip_baseline
```

V-JEPA baseline and transition ablations. Each variant trains the shared
retrieval model on a different view of the cached V-JEPA states, then evaluates
it. `pooled` is the V-JEPA baseline without transitions (B2), and
`all_ablations` runs `no_transition`, `mean_transition`, `final_minus_initial`,
and `full_transition_without_query_conditioning` (B3):

```bash
python -m src.models.baselines.vjepa_baseline --variant pooled all_ablations
```

To write rankings for the main model, evaluate its checkpoint with
`--variant full_transition` (it reads
`outputs/checkpoints/transition_model_best.pt` by default and skips training).
Pass `--checkpoint <path>` to evaluate any other saved variant without
retraining.

Categorize retrieval failures, compare models, and save example figures:

```bash
python -m src.models.baselines.error_analysis --figure-dir outputs/results/qualitative
```

Run tests with `python -m pytest`. See [Tickets](docs/Tickets.md) for interfaces,
[PRD](docs/PRD.md) for project details, and [Owners](docs/OWNERS.md) for team responsibilities.
