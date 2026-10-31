"""Shared helpers for loading cached features and batching transition sequences."""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.dataset import load_annotations
from src.models.transition_model.encoder import load_video_feature_cache
from src.models.transition_model.transition import compute_transitions


# Config

FEATURE_SUFFIX = ".npz"


# Sub functions

def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: Path) -> Path:
    """Create a directory and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_device(device: str | None = None) -> str:
    """Return the requested device, or the best one available."""
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def feature_path(feature_dir: Path, video_id: str) -> Path:
    """Return the feature cache path for one video."""
    return Path(feature_dir) / f"{video_id}{FEATURE_SUFFIX}"


def load_states(feature_dir: Path, video_id: str) -> np.ndarray:
    """Load cached state embeddings (T, D) for one video."""
    return load_video_feature_cache(feature_path(feature_dir, video_id))["frame_embeddings"]


def load_transitions(feature_dir: Path, video_id: str) -> np.ndarray:
    """Load cached states for one video and return its transitions (T - 1, D)."""
    return compute_transitions(load_states(feature_dir, video_id))


def pad_sequences(sequences: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Pad (L_i, D) arrays to (N, L_max, D); the mask is True for real steps."""
    if not sequences:
        raise ValueError("sequences must not be empty")
    max_length = max(len(sequence) for sequence in sequences)
    dim = sequences[0].shape[1]
    padded = np.zeros((len(sequences), max_length, dim), dtype=np.float32)
    mask = np.zeros((len(sequences), max_length), dtype=bool)
    for row, sequence in enumerate(sequences):
        padded[row, : len(sequence)] = sequence
        mask[row, : len(sequence)] = True
    return padded, mask


class TransitionDataset(Dataset):
    """Yield (transitions, label) pairs, reading feature caches lazily."""

    def __init__(self, video_ids: list[str], labels: list[int], feature_dir: Path):
        if len(video_ids) != len(labels):
            raise ValueError("video_ids and labels must have the same length")
        self.video_ids = video_ids
        self.labels = labels
        self.feature_dir = Path(feature_dir)

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        return load_transitions(self.feature_dir, self.video_ids[index]), self.labels[index]


def collate_transitions(
    batch: list[tuple[np.ndarray, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return padded transitions (B, L, D), mask (B, L), and labels (B,)."""
    sequences, labels = zip(*batch)
    padded, mask = pad_sequences(list(sequences))
    return (
        torch.from_numpy(padded),
        torch.from_numpy(mask),
        torch.tensor(labels, dtype=torch.long),
    )


# Main function

def load_split_rows(
    annotations: Path,
    feature_dir: Path,
    limit: int | None = None,
) -> tuple[list[dict], int]:
    """Return rows whose target video has cached features, plus the skipped count."""
    rows = load_annotations(Path(annotations))
    kept = [
        row
        for row in rows
        if feature_path(feature_dir, str(row["target_video_id"])).exists()
    ]
    skipped = len(rows) - len(kept)
    return (kept[:limit] if limit is not None else kept), skipped
