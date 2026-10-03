# Accumulated Temporal Transitions for Text-Video Retrieval

This project studies whether explicit latent temporal changes improve text-to-video retrieval on Something-Something V2. The core idea is to encode ordered video states, compute consecutive transitions, and compare transition-aware video representations against text queries. The first milestone is a shared skeleton so data, encoder, baseline, and evaluation work can proceed independently.

## Folder Structure

```text
project_root/
├── docs/
│   ├── OWNERS.md
│   ├── PRD.md
│   └── Tickets.md
├── README.md
├── config.py
├── requirements.txt
├── src/
│   ├── data/
│   │   └── dataset.py
│   ├── models/
│   │   ├── transition_model/
│   │   │   ├── encoder.py
│   │   │   ├── transition.py
│   │   │   ├── retrieval.py
│   │   │   ├── utils.py
│   │   │   └── pipelines/
│   │   │       ├── extract_features.py
│   │   │       ├── train.py
│   │   │       └── evaluate.py
│   │   └── baselines/
│   │   │   ├── clip_baseline.py
│   │   │   └── vjepa_baseline.py
│   └── eval/
│       └── evaluate.py
└── outputs/
    ├── features/
    ├── checkpoints/
    └── results/
```

## Setup

Install and initialize [pyenv](https://github.com/pyenv/pyenv#installation) first.
If the old `.venv` is active, run `deactivate`; if Conda `(base)` is active,
run `conda deactivate`. Remove an existing Python 3.13 `.venv` with
`rm -rf .venv` before setup. On macOS 12 Intel, use Python 3.11 and the
compatible binary packages below:

```bash
pyenv install 3.11
pyenv local 3.11
pyenv exec python -m venv .venv
source .venv/bin/activate
python --version
python -m pip install --only-binary=:all: \
  "numpy<2" "opencv-python==4.10.0.84" \
  "torch==2.2.2" "torchvision==0.17.2" \
  -r requirements.txt
```

The setup command installs `pytest` from `requirements.txt`. Run the tests with
the active environment's Python:

```bash
python -m pytest
```

Python source files expose the shared interfaces from `docs/Tickets.md` so each
owner can fill in their part independently.

## Expected Data Format

Dataset loading expects split metadata files under `data/annotations/` by default:

```text
data/
├── annotations/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
└── videos/
    └── <video_id>.webm
```

Each JSONL row should contain:

```json
{
  "query_id": "q_000001",
  "video_id": "123",
  "video_path": "data/videos/123.webm",
  "text_query": "putting something on something",
  "target_video_id": "123",
  "split": "train"
}
```

CSV files with the same column names are also supported. Local paths should be relative to the repository or configured through environment variables. Do not commit full datasets, cached features, checkpoints, or result dumps.

## Team Starting Points

Quynh primarily owns `config.py` and `src/models/transition_model/`.

Priyanka primarily owns `src/data/`, `src/eval/`, split preparation, hard-negative metadata, and dataset statistics.

Hao primarily owns `src/models/baselines/` and baseline runs through the shared evaluator.

See `docs/OWNERS.md` before editing across another teammate's folder.

## Quynh milestone 1: encoders and feature caches

Install dependencies with the setup command above. The V-JEPA 2 checkpoint is large and downloads into `outputs/model_cache/` by default. Mock mode does not load model weights.

```bash
python -m src.models.transition_model.pipelines.extract_features \
  --annotations data/annotations/train.jsonl \
  --output-dir outputs/features/vjepa \
  --num-frames 16 --limit 10
```

The extractor reads unique `video_id` values in annotation order, samples ordered frames through `src.data.dataset`, and writes one `.npz` per video. Each cache contains `video_id`, `frame_indices`, `(T, D)` `frame_embeddings`, a `(D,)` mean `pooled_embedding`, and `encoder_name`. It skips matching caches on rerun; use `--overwrite` after changing sampling or source videos. The default pretrained encoder is [`facebook/vjepa2-vitl-fpc64-256`](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256). Its output consists of spatial patches grouped into temporal tubelets. The extractor averages spatial patches at each tubelet, then interpolates those ordered states to the sampled frame count. These are contextualized states, not independently encoded still images.

For a small local interface check without downloading model weights:

```bash
python -m src.models.transition_model.pipelines.extract_features \
  --annotations data/annotations/train.jsonl \
  --output-dir outputs/features/mock \
  --encoder mock --limit 2
```

The mock video embeddings are RGB channel means and standard deviations. They are only useful for checking data flow; do not use them as V-JEPA results.

The shared text encoder defaults to [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It returns normalized embeddings in query order:

```python
from src.models.transition_model.encoder import encode_text_queries
queries = encode_text_queries(["picking up a cup", "putting down a cup"])
```

Text and V-JEPA embeddings have different dimensions and are **not aligned**. The milestone 2 model must learn a projection or matching function before scoring them. The `encoder_name="mock"` text option is a deterministic interface check only. The CLIP baseline uses its own CLIP text encoder.

All configured paths accept absolute values or paths relative to the repository root: `ATTVR_DATA_ROOT`, `ATTVR_ANNOTATION_ROOT`, `ATTVR_VIDEO_ROOT`, `ATTVR_OUTPUT_ROOT`, `ATTVR_FEATURE_ROOT`, `ATTVR_CHECKPOINT_ROOT`, `ATTVR_RESULT_ROOT`, and `ATTVR_MODEL_CACHE_ROOT`. The model IDs can be overridden with `ATTVR_VJEPA_MODEL_NAME` and `ATTVR_TEXT_MODEL_NAME`. `DEFAULT_NUM_FRAMES` is 16 and `RANDOM_SEED` is 42 unless set through their matching `ATTVR_` variables.
