"""Transition-aware retrieval model with a query-conditioned reranker.

Each text query is projected into the transition space. A video's score is the
scaled cosine similarity plus a reranker logit computed from the interaction of
that specific query and video. Setting query_conditioned=False drops the
reranker, which gives the "reranker without query conditioning" ablation.
"""

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.models.transition_model.transition import (
    DROPOUT,
    MODEL_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    TransitionEncoder,
)


# Config

RERANK_HIDDEN_DIM = 512
INITIAL_TEMPERATURE = 0.07
SCORE_CHUNK_SIZE = 256


# Sub functions

class TransitionRetrievalModel(nn.Module):
    """Score text queries against transition-encoded videos."""

    def __init__(
        self,
        video_dim: int,
        text_dim: int,
        model_dim: int = MODEL_DIM,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
        rerank_hidden_dim: int = RERANK_HIDDEN_DIM,
        query_conditioned: bool = True,
    ):
        super().__init__()
        self.config = {
            "video_dim": video_dim,
            "text_dim": text_dim,
            "model_dim": model_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "dropout": dropout,
            "rerank_hidden_dim": rerank_hidden_dim,
            "query_conditioned": query_conditioned,
        }
        self.video_encoder = TransitionEncoder(
            video_dim, model_dim, num_heads, num_layers, dropout
        )
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, model_dim), nn.LayerNorm(model_dim)
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / INITIAL_TEMPERATURE)))
        self.reranker = None
        if query_conditioned:
            self.reranker = nn.Sequential(
                nn.Linear(4 * model_dim, rerank_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(rerank_hidden_dim, 1),
            )

    def encode_videos(self, transitions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized video representations (B, model_dim)."""
        return F.normalize(self.video_encoder(transitions, mask), dim=-1)

    def encode_texts(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized text representations (N, model_dim)."""
        return F.normalize(self.text_projection(text_embeddings), dim=-1)

    def score(self, texts: torch.Tensor, videos: torch.Tensor) -> torch.Tensor:
        """Score encoded texts (Nq, d) against encoded videos (Nc, d) -> (Nq, Nc)."""
        scores = self.logit_scale.exp().clamp(max=100) * texts @ videos.T
        if self.reranker is None:
            return scores
        text_grid = texts[:, None, :].expand(-1, videos.shape[0], -1)
        video_grid = videos[None, :, :].expand(texts.shape[0], -1, -1)
        pairs = torch.cat(
            (text_grid, video_grid, text_grid * video_grid, (text_grid - video_grid).abs()),
            dim=-1,
        )
        return scores + self.reranker(pairs).squeeze(-1)


def save_checkpoint(
    model: TransitionRetrievalModel,
    path: Path,
    **metadata,
) -> Path:
    """Save model weights, architecture config, and extra metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_config": model.config, "state_dict": model.state_dict(), **metadata},
        path,
    )
    return path


def load_checkpoint(
    path: Path,
    device: str = "cpu",
) -> tuple[TransitionRetrievalModel, dict]:
    """Rebuild a model from a checkpoint and return it with the checkpoint dict."""
    checkpoint = torch.load(Path(path), map_location=device)
    model = TransitionRetrievalModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


# Main functions

def score_transition_retrieval(
    text_embeddings: np.ndarray,
    video_representations: np.ndarray,
    model: TransitionRetrievalModel | None = None,
    chunk_size: int = SCORE_CHUNK_SIZE,
) -> np.ndarray:
    """Return score matrix with shape (N_queries, N_candidates).

    Without a model, scores are dot products of same-sized embeddings. With a
    model, text_embeddings are raw text-encoder outputs and
    video_representations come from model.encode_videos.
    """
    text_embeddings = np.asarray(text_embeddings, dtype=np.float32)
    video_representations = np.asarray(video_representations, dtype=np.float32)
    if text_embeddings.ndim != 2 or video_representations.ndim != 2:
        raise ValueError("embeddings must be 2D arrays")

    if model is None:
        if text_embeddings.shape[1] != video_representations.shape[1]:
            raise ValueError("text and video embedding dimensions must match")
        return (text_embeddings @ video_representations.T).astype(np.float32)

    # Query-video pair features grow as Nq x Nc x 4d, so score in tiles.
    model.eval()
    device = next(model.parameters()).device
    scores = np.empty((len(text_embeddings), len(video_representations)), dtype=np.float32)
    with torch.inference_mode():
        texts = model.encode_texts(torch.from_numpy(text_embeddings).to(device))
        videos = torch.from_numpy(video_representations).to(device)
        for q in range(0, len(texts), chunk_size):
            for c in range(0, len(videos), chunk_size):
                tile = model.score(texts[q : q + chunk_size], videos[c : c + chunk_size])
                scores[q : q + chunk_size, c : c + chunk_size] = tile.float().cpu().numpy()
    return scores


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
