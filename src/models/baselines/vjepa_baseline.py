import numpy as np


def score_vjepa_baseline(
    text_embeddings: np.ndarray,
    video_embeddings: np.ndarray,
) -> np.ndarray:
    """Return score matrix with shape (N_queries, N_candidates)."""
    text_embeddings = np.asarray(text_embeddings, dtype=np.float32)
    video_embeddings = np.asarray(video_embeddings, dtype=np.float32)
    if text_embeddings.ndim != 2 or video_embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D arrays")
    if text_embeddings.shape[1] != video_embeddings.shape[1]:
        raise ValueError("text and video embedding dimensions must match")
    return (text_embeddings @ video_embeddings.T).astype(np.float32)


def run_transition_ablation(
    variant: str,
    text_embeddings: np.ndarray,
    state_embeddings: dict[str, np.ndarray],
    transition_embeddings: dict[str, np.ndarray],
) -> np.ndarray:
    """Return score matrix for one ablation variant."""
    raise NotImplementedError(
        "Ablation runners should call shared transition/retrieval helpers once "
        "the transition model interface is finalized."
    )
