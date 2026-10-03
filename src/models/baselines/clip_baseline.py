import numpy as np


def encode_clip_text_queries(texts: list[str]) -> np.ndarray:
    """Return CLIP text embeddings with shape (N, D), dtype float32."""
    if any(not text.strip() for text in texts):
        raise ValueError("texts must not contain empty strings")
    raise NotImplementedError("Connect this function to CLIP or open_clip_torch.")


def encode_clip_frames(frames: np.ndarray) -> np.ndarray:
    """Return CLIP frame embeddings with shape (T, D), dtype float32."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must have shape (T, H, W, 3)")
    raise NotImplementedError("Connect this function to CLIP or open_clip_torch.")


def pool_frame_embeddings(frame_embeddings: np.ndarray) -> np.ndarray:
    """Return one video embedding with shape (D,), dtype float32."""
    frame_embeddings = np.asarray(frame_embeddings, dtype=np.float32)
    if frame_embeddings.ndim != 2:
        raise ValueError("frame_embeddings must have shape (T, D)")
    if frame_embeddings.shape[0] == 0:
        raise ValueError("frame_embeddings must not be empty")
    return frame_embeddings.mean(axis=0).astype(np.float32)


def score_clip_retrieval(
    text_embeddings: np.ndarray,
    video_embeddings: np.ndarray,
) -> np.ndarray:
    """Return retrieval scores with shape (N_queries, N_candidates)."""
    text_embeddings = np.asarray(text_embeddings, dtype=np.float32)
    video_embeddings = np.asarray(video_embeddings, dtype=np.float32)
    if text_embeddings.ndim != 2 or video_embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D arrays")
    if text_embeddings.shape[1] != video_embeddings.shape[1]:
        raise ValueError("text and video embedding dimensions must match")
    return (text_embeddings @ video_embeddings.T).astype(np.float32)
