"""Latent transitions z[t + 1] - z[t] and the temporal encoder that summarizes them."""

from pathlib import Path

import numpy as np
import torch
from torch import nn


# Config

MODEL_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1
MAX_TRANSITIONS = 64


# Sub functions

def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average (B, L, D) over valid steps, where mask (B, L) is True."""
    weights = mask.unsqueeze(-1).to(values.dtype)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)


class TransitionEncoder(nn.Module):
    """Temporal self-attention over ordered transitions, pooled to one vector."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int = MODEL_DIM,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
        max_transitions: int = MAX_TRANSITIONS,
    ):
        super().__init__()
        self.max_transitions = max_transitions
        self.input = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, model_dim))
        self.positions = nn.Embedding(max_transitions, model_dim)
        layer = nn.TransformerEncoderLayer(
            model_dim,
            num_heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers, enable_nested_tensor=False
        )
        self.output = nn.LayerNorm(model_dim)

    def forward(self, transitions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Map transitions (B, L, D) and mask (B, L) to representations (B, model_dim)."""
        length = transitions.shape[1]
        if length > self.max_transitions:
            raise ValueError(f"at most {self.max_transitions} transitions are supported")
        positions = torch.arange(length, device=transitions.device)
        tokens = self.input(transitions) + self.positions(positions)
        tokens = self.encoder(tokens, src_key_padding_mask=~mask)
        return self.output(masked_mean(tokens, mask))


def save_transition_feature_cache(
    video_id: str,
    state_embeddings: np.ndarray,
    transition_representation: np.ndarray,
    output_dir: Path,
) -> Path:
    """Save states, transitions, and the encoded representation for one video."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_id}.npz"
    state_embeddings = np.asarray(state_embeddings, dtype=np.float32)
    np.savez_compressed(
        output_path,
        video_id=np.array(video_id),
        state_embeddings=state_embeddings,
        transition_embeddings=compute_transitions(state_embeddings),
        transition_representation=np.asarray(transition_representation, dtype=np.float32),
    )
    return output_path


def load_transition_feature_cache(path: Path) -> dict:
    """Load one transition cache file."""
    with np.load(Path(path), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


# Main functions

def compute_transitions(state_embeddings: np.ndarray) -> np.ndarray:
    """Return z[t + 1] - z[t] with shape (T - 1, D), dtype float32."""
    state_embeddings = np.asarray(state_embeddings, dtype=np.float32)
    if state_embeddings.ndim != 2:
        raise ValueError("state_embeddings must have shape (T, D)")
    if state_embeddings.shape[0] < 2:
        raise ValueError("at least two state embeddings are required")
    return np.diff(state_embeddings, axis=0).astype(np.float32)


def encode_transition_sequence(
    transition_embeddings: np.ndarray,
    mask: np.ndarray | None = None,
    encoder: TransitionEncoder | None = None,
) -> np.ndarray:
    """Return a transition-aware video representation with shape (D_model,).

    Without an encoder, valid transitions are mean-pooled (early-integration
    fallback). With an encoder, it runs in eval mode, so the output is
    deterministic.
    """
    transition_embeddings = np.asarray(transition_embeddings, dtype=np.float32)
    if transition_embeddings.ndim != 2 or transition_embeddings.shape[0] == 0:
        raise ValueError("transition_embeddings must have nonempty shape (T - 1, D)")
    if mask is None:
        mask = np.ones(transition_embeddings.shape[0], dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (transition_embeddings.shape[0],):
        raise ValueError("mask must have shape (T - 1,)")
    if not mask.any():
        raise ValueError("mask must select at least one transition")

    if encoder is None:
        return transition_embeddings[mask].mean(axis=0).astype(np.float32)

    encoder.eval()
    device = next(encoder.parameters()).device
    with torch.inference_mode():
        representation = encoder(
            torch.from_numpy(transition_embeddings[None]).to(device),
            torch.from_numpy(mask[None]).to(device),
        )
    return representation[0].float().cpu().numpy()
