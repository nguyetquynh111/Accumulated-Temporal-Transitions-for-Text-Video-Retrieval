"""Shared project paths and model defaults.

Relative environment paths are interpreted from the repository root, so commands
behave the same way regardless of the current working directory.
"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    path = Path(value).expanduser() if value else default
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


DATA_ROOT = path_from_env("ATTVR_DATA_ROOT", PROJECT_ROOT / "data")
ANNOTATION_ROOT = path_from_env("ATTVR_ANNOTATION_ROOT", DATA_ROOT / "annotations")
VIDEO_ROOT = path_from_env("ATTVR_VIDEO_ROOT", DATA_ROOT / "videos")

OUTPUT_ROOT = path_from_env("ATTVR_OUTPUT_ROOT", PROJECT_ROOT / "outputs")
FEATURE_ROOT = path_from_env("ATTVR_FEATURE_ROOT", OUTPUT_ROOT / "features")
CHECKPOINT_ROOT = path_from_env("ATTVR_CHECKPOINT_ROOT", OUTPUT_ROOT / "checkpoints")
RESULT_ROOT = path_from_env("ATTVR_RESULT_ROOT", OUTPUT_ROOT / "results")
MODEL_CACHE_ROOT = path_from_env("ATTVR_MODEL_CACHE_ROOT", OUTPUT_ROOT / "model_cache")

DEFAULT_NUM_FRAMES = int(os.environ.get("ATTVR_DEFAULT_NUM_FRAMES", "16"))
RANDOM_SEED = int(os.environ.get("ATTVR_RANDOM_SEED", "42"))
VJEPA_MODEL_NAME = os.environ.get(
    "ATTVR_VJEPA_MODEL_NAME", "facebook/vjepa2-vitl-fpc64-256"
)
TEXT_MODEL_NAME = os.environ.get(
    "ATTVR_TEXT_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)


def ensure_output_dirs() -> None:
    """Create directories used for generated artifacts and model downloads."""
    for path in (FEATURE_ROOT, CHECKPOINT_ROOT, RESULT_ROOT, MODEL_CACHE_ROOT):
        path.mkdir(parents=True, exist_ok=True)
