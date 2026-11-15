"""Shared project paths and model defaults.

Relative environment paths are interpreted from the repository root, so commands
behave the same way regardless of the current working directory.
"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def path_from_env(name: str, default: Path) -> Path:
    """Return the path in environment variable `name`, or `default` when unset."""
    value = os.environ.get(name)
    path = Path(value).expanduser() if value else default
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def int_from_env(name: str, default: int) -> int:
    """Return the integer in environment variable `name`, or `default` when unset."""
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


DATA_ROOT = path_from_env("ATTVR_DATA_ROOT", PROJECT_ROOT / "data")
ANNOTATION_ROOT = path_from_env("ATTVR_ANNOTATION_ROOT", DATA_ROOT / "annotations")
VIDEO_ROOT = path_from_env("ATTVR_VIDEO_ROOT", DATA_ROOT / "videos")

OUTPUT_ROOT = path_from_env("ATTVR_OUTPUT_ROOT", PROJECT_ROOT / "outputs")
FEATURE_ROOT = path_from_env("ATTVR_FEATURE_ROOT", OUTPUT_ROOT / "features")
CHECKPOINT_ROOT = path_from_env("ATTVR_CHECKPOINT_ROOT", OUTPUT_ROOT / "checkpoints")
RESULT_ROOT = path_from_env("ATTVR_RESULT_ROOT", OUTPUT_ROOT / "results")
MODEL_CACHE_ROOT = path_from_env("ATTVR_MODEL_CACHE_ROOT", OUTPUT_ROOT / "model_cache")
# Slurm job logs and per-epoch training logs.
LOG_ROOT = path_from_env("ATTVR_LOG_ROOT", PROJECT_ROOT / "logs")

# Per-encoder cache directories from the shared feature cache contract.
VJEPA_FEATURE_ROOT = FEATURE_ROOT / "vjepa"
CLIP_FEATURE_ROOT = FEATURE_ROOT / "clip"
TRANSITION_FEATURE_ROOT = FEATURE_ROOT / "transitions"

# Frame sampling range allowed by the shared dataset contract.
MIN_NUM_FRAMES = 8
MAX_NUM_FRAMES = 16
DEFAULT_NUM_FRAMES = int_from_env("ATTVR_DEFAULT_NUM_FRAMES", 16)
RANDOM_SEED = int_from_env("ATTVR_RANDOM_SEED", 42)
VJEPA_MODEL_NAME = os.environ.get(
    "ATTVR_VJEPA_MODEL_NAME", "facebook/vjepa2-vitl-fpc64-256"
)
TEXT_MODEL_NAME = os.environ.get(
    "ATTVR_TEXT_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)

# Training protocol shared by the main model and every trained ablation (T2.4).
# Training stops once validation SELECTION_METRIC has not improved for
# EARLY_STOPPING_PATIENCE epochs, or at TRAIN_MAX_EPOCHS.
TRAIN_MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
TRAIN_BATCH_SIZE = 128
TRAIN_LEARNING_RATE = 1e-4
TRAIN_WEIGHT_DECAY = 0.01
TRAIN_SELECTION_METRIC = "mrr"
TRAIN_SEEDS = (42, 43, 44)


def ensure_output_dirs() -> None:
    """Create directories used for generated artifacts and model downloads."""
    for path in (
        VJEPA_FEATURE_ROOT,
        CLIP_FEATURE_ROOT,
        TRANSITION_FEATURE_ROOT,
        CHECKPOINT_ROOT,
        RESULT_ROOT,
        MODEL_CACHE_ROOT,
        LOG_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)
