import numpy as np


def score_transition_retrieval(
    text_embeddings: np.ndarray,
    video_representations: np.ndarray,
) -> np.ndarray:
    """Return score matrix with shape (N_queries, N_candidates)."""
    text_embeddings = np.asarray(text_embeddings, dtype=np.float32)
    video_representations = np.asarray(video_representations, dtype=np.float32)
    if text_embeddings.ndim != 2 or video_representations.ndim != 2:
        raise ValueError("embeddings must be 2D arrays")
    if text_embeddings.shape[1] != video_representations.shape[1]:
        raise ValueError("text and video embedding dimensions must match")
    return (text_embeddings @ video_representations.T).astype(np.float32)


def rank_candidates(
    scores: np.ndarray,
    candidate_video_ids: list[str],
) -> list[list[str]]:
    """Return candidate ids sorted from best to worst for each query."""
    scores = np.asarray(scores)
    if scores.ndim != 2:
        raise ValueError("scores must have shape (N_queries, N_candidates)")
    if scores.shape[1] != len(candidate_video_ids):
        raise ValueError("candidate_video_ids length must match score columns")

    ranked = []
    for row in scores:
        order = np.argsort(-row, kind="stable")
        ranked.append([candidate_video_ids[index] for index in order])
    return ranked
