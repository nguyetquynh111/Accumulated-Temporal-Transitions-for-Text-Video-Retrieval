import numpy as np


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
) -> np.ndarray:
    """Return a transition-aware video representation with shape (D_model,)."""
    transition_embeddings = np.asarray(transition_embeddings, dtype=np.float32)
    if transition_embeddings.ndim != 2:
        raise ValueError("transition_embeddings must have shape (T - 1, D)")

    if mask is None:
        if transition_embeddings.shape[0] == 0:
            raise ValueError("transition_embeddings must not be empty")
        return transition_embeddings.mean(axis=0).astype(np.float32)

    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (transition_embeddings.shape[0],):
        raise ValueError("mask must have shape (T - 1,)")
    if not mask.any():
        raise ValueError("mask must select at least one transition")
    return transition_embeddings[mask].mean(axis=0).astype(np.float32)
