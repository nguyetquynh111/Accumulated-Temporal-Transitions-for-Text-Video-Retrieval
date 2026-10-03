# Tickets - Accumulated Temporal Transitions for Text-Video Retrieval

This document defines the implementation tickets, shared data contracts, and
owner-level function inputs/outputs for the project. All owners should keep
their module interfaces compatible with the contracts below so that data,
model, baseline, and evaluation work can be developed independently.

## Shared Data Contracts

### Annotation Files

All split files must use JSONL by default. CSV is allowed only if it uses the
same column names.

Default paths:

```text
data/annotations/train.jsonl
data/annotations/val.jsonl
data/annotations/test.jsonl
data/annotations/hard_negatives.jsonl
data/videos/<video_id>.webm
```

Each row in `train.jsonl`, `val.jsonl`, and `test.jsonl` must follow this
schema:

```json
{
  "query_id": "q_000001",
  "video_id": "12345",
  "video_path": "data/videos/12345.webm",
  "text_query": "putting something on something",
  "target_video_id": "12345",
  "template": "putting [something] on [something]",
  "label": "putting something on something",
  "split": "train"
}
```

Required fields:

- `query_id`: unique string identifier for the query row.
- `video_id`: string identifier for the source video.
- `video_path`: repository-relative or absolute path to the video file.
- `text_query`: text used by retrieval models.
- `target_video_id`: video id that should be retrieved for the query.
- `split`: one of `train`, `val`, or `test`.

Optional but recommended fields:

- `template`: original Something-Something action template.
- `label`: normalized action label.
- `objects`: list of object strings when available.
- `metadata`: object containing any extra source dataset fields.

### Dataset Sample Format

Dataset and dataloader functions should return dictionaries with these keys:

```python
{
    "query_id": str,
    "video_id": str,
    "video_path": str,
    "text_query": str,
    "target_video_id": str,
    "frame_indices": np.ndarray,  # shape: (T,), dtype: int64
    "frames": np.ndarray,         # shape: (T, H, W, 3), dtype: uint8, RGB order
}
```

Frame sampling must preserve temporal order. The default number of frames is
`T = 16`; small-subset experiments may use `T = 8`.

### Feature Cache Format

Feature extraction should save one `.npz` file per video:

```text
outputs/features/vjepa/<video_id>.npz
outputs/features/clip/<video_id>.npz
```

Each `.npz` file must contain:

- `video_id`: scalar string or one-element string array.
- `frame_indices`: array with shape `(T,)`, dtype `int64`.
- `frame_embeddings`: array with shape `(T, D)`, dtype `float32`.
- `pooled_embedding`: array with shape `(D,)`, dtype `float32`.
- `encoder_name`: scalar string or one-element string array.

Transition features should be saved as:

```text
outputs/features/transitions/<video_id>.npz
```

Each transition `.npz` file must contain:

- `video_id`: scalar string or one-element string array.
- `state_embeddings`: array with shape `(T, D)`, dtype `float32`.
- `transition_embeddings`: array with shape `(T - 1, D)`, dtype `float32`.
- `transition_representation`: array with shape `(D_model,)`, dtype `float32`.

### Retrieval Score Format

Full-split models and baselines must produce scores using the same shape:

```python
scores: np.ndarray          # shape: (num_queries, num_candidates), dtype: float32
query_ids: list[str]        # length: num_queries
candidate_video_ids: list[str]  # length: num_candidates
target_video_ids: list[str] # length: num_queries
```

Higher scores must mean better matches.

Hard-negative evaluation may use a different candidate list for each query. In
that case, each hard-negative row must carry its own `candidate_video_ids`, and
the evaluator should score each row against only those candidates while
returning the same metric names.

### Metrics Format

Evaluation functions must return:

```python
{
    "recall@1": float,
    "recall@5": float,
    "recall@10": float,
    "mrr": float,
    "map": float,
    "num_queries": int,
    "num_candidates": int,
}
```

## Milestone 1 - Data and Basic Infrastructure

### T1.1 - Prepare Split Metadata

Owner: Priyanka

Files:

- `src/data/dataset.py`
- `data/annotations/*.jsonl` generated locally only
- `outputs/results/dataset_stats.json` generated locally only

Inputs:

- Raw Something-Something V2 metadata.
- Local video directory path.
- Split configuration from `config.py`.

Required functions:

```python
def load_annotations(path: Path) -> list[dict]:
    """Load JSONL or CSV annotation rows using the shared annotation schema."""

def create_splits(
    raw_metadata_path: Path,
    video_root: Path,
    output_dir: Path,
    seed: int,
) -> dict[str, Path]:
    """Create train, val, and test split files and return their paths."""

def compute_dataset_stats(annotation_paths: dict[str, Path]) -> dict:
    """Compute counts by split, label, and missing-video status."""
```

Outputs:

- `train.jsonl`, `val.jsonl`, and `test.jsonl` in the shared annotation format.
- `dataset_stats.json` with counts by split and label.

Acceptance criteria:

- Every row has `query_id`, `video_id`, `video_path`, `text_query`,
  `target_video_id`, and `split`.
- Split generation is deterministic for the same seed.
- Missing video files are reported in stats instead of failing silently.
- The held-out test set is created from labeled validation data, not from the
  official hidden-label test set.

### T1.2 - Implement Dataset Loading and Ordered Frame Sampling

Owner: Priyanka

Files:

- `src/data/dataset.py`

Inputs:

- One split file in JSONL or CSV format.
- `num_frames` in the range 8-16.
- Optional video transform callable.

Required functions:

```python
def sample_ordered_frame_indices(
    total_frames: int,
    num_frames: int,
) -> np.ndarray:
    """Return ordered frame indices with shape (num_frames,)."""

def load_video_frames(
    video_path: Path,
    frame_indices: np.ndarray,
) -> np.ndarray:
    """Return RGB frames with shape (T, H, W, 3), dtype uint8."""

def get_dataset_sample(
    row: dict,
    num_frames: int,
) -> dict:
    """Return one sample using the shared dataset sample format."""
```

Outputs:

- Dataset samples matching the shared sample format.
- Ordered RGB frame arrays ready for encoders and baselines.

Acceptance criteria:

- Sampling preserves chronological order.
- Short videos are handled by repeated or nearest valid frame indices.
- Output frames use RGB channel order and `uint8` dtype.
- The same input video and frame count produce the same sampled indices.

### T1.3 - Set Up Shared Configuration and Paths

Owner: Quynh

Files:

- `config.py`
- `README.md` if command examples need updates

Inputs:

- Repository root.
- Optional environment variables for data, outputs, and model cache paths.

Required functions or constants:

```python
PROJECT_ROOT: Path
DATA_ROOT: Path
ANNOTATION_ROOT: Path
VIDEO_ROOT: Path
OUTPUT_ROOT: Path
FEATURE_ROOT: Path
CHECKPOINT_ROOT: Path
RESULT_ROOT: Path
DEFAULT_NUM_FRAMES: int
RANDOM_SEED: int
```

Outputs:

- Centralized paths and defaults imported by all modules.

Acceptance criteria:

- No owner hard-codes absolute local paths in project modules.
- Required output directories can be created from config values.
- Defaults match the shared data contracts in this document.

### T1.4 - Implement V-JEPA 2 Feature Extraction Interface

Owner: Quynh

Files:

- `src/models/transition_model/encoder.py`
- `src/models/transition_model/pipelines/extract_features.py`
- `outputs/features/vjepa/*.npz` generated locally only

Inputs:

- Dataset samples from Priyanka's loader.
- Pretrained V-JEPA 2 model or a temporary mock encoder for early integration.

Required functions:

```python
def encode_video_frames(frames: np.ndarray) -> np.ndarray:
    """Return frame/state embeddings with shape (T, D), dtype float32."""

def save_video_feature_cache(
    video_id: str,
    frame_indices: np.ndarray,
    frame_embeddings: np.ndarray,
    output_dir: Path,
    encoder_name: str,
) -> Path:
    """Save one feature cache file and return its path."""

def load_video_feature_cache(path: Path) -> dict:
    """Load one feature cache file using the shared feature format."""
```

Outputs:

- V-JEPA feature cache files in the shared `.npz` format.

Acceptance criteria:

- The extraction pipeline can run on a small subset without changing dataset
  or evaluation code.
- Cached `frame_embeddings` preserve frame order.
- `pooled_embedding` is available for the V-JEPA non-transition baseline.

### T1.5 - Implement Text Encoder Interface

Owner: Quynh

Files:

- `src/models/transition_model/encoder.py`

Inputs:

- List of text queries.
- Text encoder name or configuration.

Required function:

```python
def encode_text_queries(texts: list[str]) -> np.ndarray:
    """Return text embeddings with shape (N, D_text), dtype float32."""
```

Outputs:

- Text embeddings aligned with input query order.

Acceptance criteria:

- Empty strings are rejected with a clear error.
- Output row `i` always corresponds to input text `i`.
- The shared text encoder is intended for the transition model and V-JEPA-style
  retrieval when dimensions are compatible.
- The CLIP baseline should use its own CLIP text encoder so CLIP text and video
  embedding dimensions always match.

### T1.6 - Implement CLIP Frame-Averaging Baseline

Owner: Hao

Files:

- `src/models/baselines/clip_baseline.py`
- `outputs/features/clip/*.npz` generated locally only
- `outputs/results/clip_baseline_small.json` generated locally only

Inputs:

- Dataset samples from Priyanka's loader.
- Text queries and candidate videos from the same split.

Required functions:

```python
def encode_clip_text_queries(texts: list[str]) -> np.ndarray:
    """Return CLIP text embeddings with shape (N, D), dtype float32."""

def encode_clip_frames(frames: np.ndarray) -> np.ndarray:
    """Return CLIP frame embeddings with shape (T, D), dtype float32."""

def pool_frame_embeddings(frame_embeddings: np.ndarray) -> np.ndarray:
    """Return one video embedding with shape (D,), dtype float32."""

def score_clip_retrieval(
    text_embeddings: np.ndarray,
    video_embeddings: np.ndarray,
) -> np.ndarray:
    """Return retrieval scores with shape (N_queries, N_candidates)."""
```

Outputs:

- CLIP score matrix using the shared retrieval score format.
- Small-subset result file for integration testing.

Acceptance criteria:

- The baseline ranks candidates with higher-is-better scores.
- Frame averaging is the only temporal pooling in this baseline.
- The output can be passed directly to Priyanka's evaluator.

## Milestone 2 - Main Model and Evaluation

### T2.1 - Implement Latent Transition Computation

Owner: Quynh

Files:

- `src/models/transition_model/transition.py`

Inputs:

- V-JEPA state embeddings with shape `(T, D)`.

Required function:

```python
def compute_transitions(state_embeddings: np.ndarray) -> np.ndarray:
    """Return z[t + 1] - z[t] with shape (T - 1, D), dtype float32."""
```

Outputs:

- Ordered transition embeddings with shape `(T - 1, D)`.

Acceptance criteria:

- The function validates that `T >= 2`.
- Output order matches the original frame/state order.
- No pooling is applied inside this function.

### T2.2 - Implement Transition Sequence Encoder

Owner: Quynh

Files:

- `src/models/transition_model/transition.py`
- `src/models/transition_model/utils.py`

Inputs:

- Transition embeddings with shape `(T - 1, D)`.
- Optional mask for padded transitions.

Required function:

```python
def encode_transition_sequence(
    transition_embeddings: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return a transition-aware video representation with shape (D_model,)."""
```

Outputs:

- One transition representation per video.

Acceptance criteria:

- The encoder supports variable sequence length after frame sampling.
- A mean-pooling fallback exists for early integration.
- The output is deterministic in evaluation mode.

### T2.3 - Implement Query-Conditioned Reranker

Owner: Quynh

Files:

- `src/models/transition_model/retrieval.py`

Inputs:

- Text embeddings with shape `(N_queries, D_text)`.
- Transition video representations with shape `(N_candidates, D_model)`.
- Candidate video ids.

Required functions:

```python
def score_transition_retrieval(
    text_embeddings: np.ndarray,
    video_representations: np.ndarray,
) -> np.ndarray:
    """Return score matrix with shape (N_queries, N_candidates)."""

def rank_candidates(
    scores: np.ndarray,
    candidate_video_ids: list[str],
) -> list[list[str]]:
    """Return candidate ids sorted from best to worst for each query."""
```

Outputs:

- Score matrix using the shared retrieval score format.
- Ranked candidate lists for qualitative analysis.

Acceptance criteria:

- Higher scores always mean better query-video matches.
- Query order and candidate order are preserved in all outputs.
- The reranker can run with cached features without re-encoding videos.

### T2.4 - Build Main Training Pipeline

Owner: Quynh

Files:

- `src/models/transition_model/pipelines/train.py`
- `outputs/checkpoints/*.pt` generated locally only

Inputs:

- Train and validation annotation files.
- Cached V-JEPA feature files.
- Text queries.

Required function:

```python
def train_transition_model(
    train_annotations: Path,
    val_annotations: Path,
    feature_dir: Path,
    checkpoint_dir: Path,
) -> Path:
    """Train the transition-aware model and return the best checkpoint path."""
```

Outputs:

- Model checkpoint.
- Training log with loss and validation metrics.

Acceptance criteria:

- The training pipeline reads cached video features instead of requiring video
  decoding every epoch.
- Validation uses the shared evaluator.
- The best checkpoint is selected by validation Recall@1 or MRR.

### T2.5 - Implement Shared Retrieval Evaluator

Owner: Priyanka

Files:

- `src/eval/evaluate.py`

Inputs:

- Score matrix with shape `(N_queries, N_candidates)`.
- Ordered `query_ids`.
- Ordered `candidate_video_ids`.
- Ordered `target_video_ids`.

Required functions:

```python
def compute_retrieval_metrics(
    scores: np.ndarray,
    query_ids: list[str],
    candidate_video_ids: list[str],
    target_video_ids: list[str],
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict:
    """Return Recall@K, MRR, mAP, and count metadata."""

def save_metrics(metrics: dict, output_path: Path) -> None:
    """Save metrics as JSON."""
```

Outputs:

- Metrics dictionary using the shared metrics format.
- JSON result files under `outputs/results/`.

Acceptance criteria:

- Metrics are correct for small hand-written examples.
- The evaluator raises a clear error when a target id is missing from the
  candidate list.
- The same evaluator supports CLIP, V-JEPA, transition model, and ablations.

### T2.6 - Build Hard-Negative Evaluation Set

Owner: Priyanka

Files:

- `src/eval/evaluate.py`
- `data/annotations/hard_negatives.jsonl` generated locally only

Inputs:

- Validation/test annotations.
- Action labels, templates, and object metadata when available.

Required function:

```python
def build_hard_negative_set(
    annotations: list[dict],
    output_path: Path,
) -> Path:
    """Create a hard-negative JSONL file and return its path."""
```

Recommended evaluator helper:

```python
def compute_hard_negative_metrics(
    hard_negative_rows: list[dict],
    score_lookup: dict[tuple[str, str], float],
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict:
    """Evaluate per-query hard-negative candidate sets."""
```

Hard-negative row schema:

```json
{
  "query_id": "q_000001",
  "target_video_id": "12345",
  "candidate_video_ids": ["12345", "67890", "24680"],
  "negative_types": ["opposite_action", "same_objects"]
}
```

Outputs:

- Hard-negative candidate sets with one positive and multiple difficult
  negatives per query.

Acceptance criteria:

- Every hard-negative row includes the target video id in `candidate_video_ids`.
- Negative examples prioritize same-object or opposite-direction actions.
- The hard-negative set can be evaluated by the same metrics function.
- Hard-negative evaluation does not assume that all queries share one global
  candidate list.

### T2.7 - Implement V-JEPA Non-Transition Baseline

Owner: Hao

Files:

- `src/models/baselines/vjepa_baseline.py`
- `outputs/results/vjepa_baseline.json` generated locally only

Inputs:

- Cached V-JEPA `pooled_embedding` vectors.
- Text embeddings.
- Candidate video ids.

Required function:

```python
def score_vjepa_baseline(
    text_embeddings: np.ndarray,
    video_embeddings: np.ndarray,
) -> np.ndarray:
    """Return score matrix with shape (N_queries, N_candidates)."""
```

Outputs:

- V-JEPA baseline score matrix and metrics.

Acceptance criteria:

- This baseline does not use explicit transition embeddings.
- It uses the same candidate ordering and evaluator as the main model.
- Results are saved in the same JSON format as the CLIP baseline.

## Milestone 3 - Experiments and Ablations

### T3.1 - Run Full Transition Model Experiments

Owner: Quynh

Files:

- `src/models/transition_model/pipelines/evaluate.py`
- `outputs/results/transition_model_full.json` generated locally only

Inputs:

- Best transition model checkpoint.
- Test split annotations.
- Cached V-JEPA and transition features.

Outputs:

- Full retrieval metrics on the held-out test split.
- Hard-negative retrieval metrics when hard-negative metadata is available.

Acceptance criteria:

- Results include Recall@1, Recall@5, Recall@10, MRR, and mAP.
- The command/config used to generate the result is documented.
- Result files include model name, checkpoint path, split name, and timestamp.

### T3.2 - Run Shared Evaluation and Comparison Tables

Owner: Priyanka

Files:

- `src/eval/evaluate.py`
- `outputs/results/comparison_table.csv` generated locally only

Inputs:

- Result JSON files from CLIP, V-JEPA, transition model, and ablations.

Required function:

```python
def build_comparison_table(
    result_paths: list[Path],
    output_path: Path,
) -> Path:
    """Create a CSV table comparing all model metrics."""
```

Outputs:

- Comparison table for final report and presentation.

Acceptance criteria:

- All models are evaluated on the same split and candidate set.
- Table columns use consistent metric names.
- Missing metrics are marked clearly instead of silently dropped.

### T3.3 - Run Ablation Experiments

Owner: Hao

Files:

- `src/models/baselines/vjepa_baseline.py`
- `src/models/baselines/clip_baseline.py` if shared scoring utilities are needed
- `outputs/results/ablation_*.json` generated locally only

Coordination note:

- Hao owns the ablation experiment runners and result files.
- Quynh owns transition-model internals, checkpoints, and query-conditioned
  scoring helpers reused by these ablations.
- Ablation code should call shared transition/retrieval helpers instead of
  duplicating transition-model logic inside baseline modules.

Inputs:

- Cached V-JEPA state embeddings.
- Cached transition embeddings.
- Text embeddings.
- Shared evaluator.

Required function:

```python
def run_transition_ablation(
    variant: str,
    text_embeddings: np.ndarray,
    state_embeddings: dict[str, np.ndarray],
    transition_embeddings: dict[str, np.ndarray],
) -> np.ndarray:
    """Return score matrix for one ablation variant."""
```

Required variants:

- `no_transition`
- `mean_transition`
- `final_minus_initial`
- `full_transition_without_query_conditioning`

Outputs:

- One metrics JSON file per ablation variant.

Acceptance criteria:

- Every ablation uses the same queries, candidate ids, and evaluator.
- Each result file records the exact ablation variant name.
- The implementation avoids duplicating evaluator logic.

## Milestone 4 - Analysis and Final Report

### T4.1 - Perform Retrieval Error Analysis

Owner: Hao

Files:

- `outputs/results/error_analysis.json` generated locally only
- Qualitative figures generated locally only

Inputs:

- Ranked retrieval outputs from CLIP, V-JEPA, and transition model.
- Test and hard-negative annotations.

Required function:

```python
def categorize_retrieval_errors(
    ranked_results: list[dict],
    annotations: dict[str, dict],
) -> list[dict]:
    """Return categorized retrieval failures for qualitative analysis."""
```

Error categories:

- `correct_objects_wrong_action`
- `wrong_temporal_direction`
- `wrong_action_ordering`
- `similar_start_end_different_motion`
- `text_ambiguity`
- `visual_ambiguity`
- `encoder_failure`

Outputs:

- Error analysis JSON.
- Qualitative examples for the final report.

Acceptance criteria:

- At least one representative example is provided for each observed category.
- Examples include query text, target video id, top retrieved video ids, and
  model names.
- Analysis compares at least one baseline against the transition model.

### T4.2 - Finalize Tables, Figures, and Reproducibility Checks

Owner: Priyanka

Files:

- `outputs/results/comparison_table.csv` generated locally only
- Final report tables and metric summaries

Inputs:

- All final metrics JSON files.
- Dataset statistics.
- Error-analysis outputs.

Outputs:

- Final metric tables.
- Dataset statistics table.
- Reproducibility checklist.

Acceptance criteria:

- Reported numbers match saved JSON results.
- Dataset counts match split files.
- Each table clearly identifies split name and evaluation setting.

### T4.3 - Integrate Methodology and Final Results

Owner: Quynh

Files:

- Final report or presentation files
- `README.md` if reproduction commands need updates

Inputs:

- Final metrics and comparison tables.
- Error-analysis examples.
- Model and ablation descriptions.

Outputs:

- Final methodology section.
- Final result summary.
- Presentation-ready model diagram and experiment summary.

Acceptance criteria:

- The report explains the transition computation `z[t + 1] - z[t]`.
- The report separates full retrieval and hard-negative retrieval results.
- Limitations include dataset scope, compute constraints, and simple transition
  operation assumptions.

## Final Deliverables

- Reproducible split metadata in the shared annotation format.
- Working dataset loader with ordered frame sampling.
- Cached V-JEPA 2 feature extraction.
- Working CLIP frame-averaging baseline.
- Working V-JEPA non-transition baseline.
- Working transition-aware retrieval model.
- Shared evaluator reporting Recall@1, Recall@5, Recall@10, MRR, and mAP.
- Hard-negative retrieval benchmark and results.
- Ablation results for the required variants.
- Error analysis with qualitative examples.
- Final presentation/report with reproducible commands and result tables.
