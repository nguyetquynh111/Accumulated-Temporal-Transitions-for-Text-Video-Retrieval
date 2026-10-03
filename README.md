# Accumulated Temporal Transitions for Text-Video Retrieval

This project explores text-to-video retrieval on Something-Something V2 using
ordered video frames and their temporal transitions. Feature extraction is
available now; training and evaluation are still in progress.

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

Run tests with `python -m pytest`. See [Tickets](docs/Tickets.md) for interfaces,
[PRD](docs/PRD.md) for project details, and [Owners](docs/OWNERS.md) for team responsibilities.
