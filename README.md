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

**Note:** The `data/` directory is excluded from Git. Team members must generate their annotation files locally from authorized official metadata.


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

Run tests with `python -m pytest`. See [Tickets](docs/Tickets.md) for interfaces,
[PRD](docs/PRD.md) for project details, and [Owners](docs/OWNERS.md) for team responsibilities.
