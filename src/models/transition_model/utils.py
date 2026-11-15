"""Shared helpers for loading cached features and batching transition sequences."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.dataset import load_annotations
from src.models.transition_model.encoder import (
    encode_text_queries,
    load_video_feature_cache,
    resolve_device,
)
from src.models.transition_model.transition import (
    DEFAULT_VIDEO_INPUT,
    build_video_tokens,
    compute_transitions,
)


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
    return resolve_device(device)


def load_jsonl(path: Path) -> list[dict]:
    """Read one JSON object per nonempty line."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def unique_in_order(values) -> list:
    """Return the distinct values in first-seen order."""
    return list(dict.fromkeys(values))


def candidate_video_ids(rows: list[dict]) -> list[str]:
    """Return the split's unique target videos, in first-seen order."""
    return unique_in_order(str(row["target_video_id"]) for row in rows)


def encode_row_texts(
    rows: list[dict],
    encoder_name: str,
    model_name: str,
    device: str | None = None,
) -> np.ndarray:
    """Encode each unique query text once and return one row per annotation."""
    texts = [row["text_query"] for row in rows]
    unique_texts = unique_in_order(texts)
    embeddings = encode_text_queries(
        unique_texts, encoder_name=encoder_name, model_name=model_name, device=device
    )
    index = {text: position for position, text in enumerate(unique_texts)}
    return embeddings[[index[text] for text in texts]]


def feature_path(feature_dir: Path, video_id: str) -> Path:
    """Return the feature cache path for one video."""
    return Path(feature_dir) / f"{video_id}{FEATURE_SUFFIX}"


def load_states(feature_dir: Path, video_id: str) -> np.ndarray:
    """Load cached state embeddings (T, D) for one video."""
    return load_video_feature_cache(feature_path(feature_dir, video_id))["frame_embeddings"]


def load_transitions(feature_dir: Path, video_id: str) -> np.ndarray:
    """Load cached states for one video and return its transitions (T - 1, D)."""
    return compute_transitions(load_states(feature_dir, video_id))


def load_video_tokens(
    feature_dir: Path,
    video_id: str,
    video_input: str = DEFAULT_VIDEO_INPUT,
) -> np.ndarray:
    """Load cached states for one video and return its encoder tokens."""
    return build_video_tokens(load_states(feature_dir, video_id), video_input)


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
    """Yield (video tokens, label) pairs, reading feature caches lazily."""

    def __init__(
        self,
        video_ids: list[str],
        labels: list[int],
        feature_dir: Path,
        video_input: str = DEFAULT_VIDEO_INPUT,
        reverse_time: bool = False,
    ):
        """With reverse_time, states are reversed before tokens are built (evaluation only)."""
        if len(video_ids) != len(labels):
            raise ValueError("video_ids and labels must have the same length")
        self.video_ids = video_ids
        self.labels = labels
        self.feature_dir = Path(feature_dir)
        self.video_input = video_input
        self.reverse_time = reverse_time

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        states = load_states(self.feature_dir, self.video_ids[index])
        if self.reverse_time:
            states = np.ascontiguousarray(states[::-1])
        return build_video_tokens(states, self.video_input), self.labels[index]


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
