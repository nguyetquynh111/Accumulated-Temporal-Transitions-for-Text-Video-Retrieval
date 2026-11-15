"""Latent transitions z[t + 1] - z[t] and the temporal encoder that summarizes them.

The main model reads states and transitions together: token t is
[z[t], z[t + 1] - z[t]]. Transitions alone drop the scene content (objects,
background) that text queries name, so the transition-only input is kept only
as an ablation.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.models.transition_model.encoder import save_npz_atomic, validate_video_id


# Config

MODEL_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1
MAX_TRANSITIONS = 64
# Video inputs the main pipeline can build from cached states:
# "state_transition" -> tokens [z[t], dz[t]], shape (T - 1, 2D) (main model)
# "transition"       -> tokens dz[t], shape (T - 1, D) (transition-only ablation)
VIDEO_INPUTS = ("state_transition", "transition")
DEFAULT_VIDEO_INPUT = "state_transition"
TRANSITION_CACHE_KEYS = (
    "video_id",
    "state_embeddings",
    "transition_embeddings",
    "transition_representation",
)


# Sub functions

def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average (B, L, D) over valid steps, where mask (B, L) is True."""
    weights = mask.unsqueeze(-1).to(values.dtype)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)


class GroupLayerNorm(nn.Module):
    """LayerNorm applied separately to consecutive feature groups, then re-joined.

    States and transitions differ in norm by about 6.5x, so one LayerNorm over
    [z, dz] would let the states dominate the projected token.
    """

    def __init__(self, group_dims: tuple[int, ...]):
        super().__init__()
        self.group_dims = tuple(int(dim) for dim in group_dims)
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for dim in self.group_dims)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        parts = values.split(self.group_dims, dim=-1)
        return torch.cat([norm(part) for norm, part in zip(self.norms, parts)], dim=-1)


class TransitionEncoder(nn.Module):
    """Temporal self-attention over ordered tokens, pooled to one vector.

    `input_groups` splits each input token into groups that are normalized
    separately, e.g. (D, D) for [z, dz] tokens. Without it, one LayerNorm covers
    the whole token, which keeps older checkpoints loadable.
    """

    def __init__(
        self,
        input_dim: int,
        model_dim: int = MODEL_DIM,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
        max_transitions: int = MAX_TRANSITIONS,
        input_groups: tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.max_transitions = max_transitions
        if input_groups is None:
            norm = nn.LayerNorm(input_dim)
        else:
            if sum(input_groups) != input_dim:
                raise ValueError("input_groups must sum to input_dim")
            norm = GroupLayerNorm(input_groups)
        self.input = nn.Sequential(norm, nn.Linear(input_dim, model_dim))
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
    validate_video_id(video_id)
    state_embeddings = np.asarray(state_embeddings, dtype=np.float32)
    transition_representation = np.asarray(transition_representation, dtype=np.float32)
    if transition_representation.ndim != 1:
        raise ValueError("transition_representation must have shape (D_model,)")
    return save_npz_atomic(
        Path(output_dir) / f"{video_id}.npz",
        video_id=np.array(video_id),
        state_embeddings=state_embeddings,
        transition_embeddings=compute_transitions(state_embeddings),
        transition_representation=transition_representation,
    )


def load_transition_feature_cache(path: Path) -> dict:
    """Load one transition cache file in the shared transition format."""
    with np.load(Path(path), allow_pickle=False) as data:
        missing = sorted(set(TRANSITION_CACHE_KEYS).difference(data.files))
        if missing:
            raise ValueError(f"transition cache missing keys: {missing}")
        return {key: data[key] for key in TRANSITION_CACHE_KEYS}


# Main functions

def compute_transitions(state_embeddings: np.ndarray) -> np.ndarray:
    """Return z[t + 1] - z[t] with shape (T - 1, D), dtype float32."""
    state_embeddings = np.asarray(state_embeddings, dtype=np.float32)
    if state_embeddings.ndim != 2:
        raise ValueError("state_embeddings must have shape (T, D)")
    if state_embeddings.shape[0] < 2:
        raise ValueError("at least two state embeddings are required")
    return np.diff(state_embeddings, axis=0).astype(np.float32)


def build_state_transition_tokens(state_embeddings: np.ndarray) -> np.ndarray:
    """Return tokens [z[t], z[t + 1] - z[t]] with shape (T - 1, 2D), dtype float32."""
    state_embeddings = np.asarray(state_embeddings, dtype=np.float32)
    transitions = compute_transitions(state_embeddings)
    return np.concatenate((state_embeddings[:-1], transitions), axis=1).astype(np.float32)


def build_video_tokens(
    state_embeddings: np.ndarray,
    video_input: str = DEFAULT_VIDEO_INPUT,
) -> np.ndarray:
    """Return the token sequence one video input type feeds to the encoder."""
    if video_input == "state_transition":
        return build_state_transition_tokens(state_embeddings)
    if video_input == "transition":
        return compute_transitions(state_embeddings)
    raise ValueError(f"video_input must be one of {VIDEO_INPUTS}, got {video_input!r}")


def encode_transition_sequence(
    transition_embeddings: np.ndarray,
    mask: np.ndarray | None = None,
    encoder: TransitionEncoder | None = None,
) -> np.ndarray:
    """Return a transition-aware video representation with shape (D_model,).

    `transition_embeddings` is the encoder's token sequence: [z, dz] tokens for
    the main model, or dz alone for the transition-only ablation.
    Without an encoder, valid transitions are mean-pooled (early-integration
    fallback). With an encoder, it runs in eval mode, so the output is
    deterministic.
    """
    transition_embeddings = np.asarray(transition_embeddings, dtype=np.float32)
    if transition_embeddings.ndim != 2 or transition_embeddings.shape[0] == 0:
        raise ValueError("transition_embeddings must have nonempty shape (L, D)")
    if mask is None:
        mask = np.ones(transition_embeddings.shape[0], dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (transition_embeddings.shape[0],):
        raise ValueError("mask must have shape (L,)")
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
