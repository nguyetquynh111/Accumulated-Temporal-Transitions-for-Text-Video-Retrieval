"""Two-stage training with a token-level cross-attention reranker (reranker B).

Same setup as train_reranker (reranker A), except for what the reranker sees:

- A (train_reranker): the video's pooled vector v, features [t, v, t*v, |t-v|].
- B (this script): the video's per-token outputs of the frozen
  TransitionEncoder. Each token comes from [z[t], dz[t]], so it carries both the
  objects (z) and the change (dz). The text attends over these tokens
  (multi-head cross-attention), which gives a query-specific video vector a,
  and the same MLP scores [t, a, t*a, |t-a|].

Everything else matches A: the stage-1 encoder (the
full_transition_without_query_conditioning checkpoint of the same seed) is
frozen, the score is the frozen scaled cosine of the pooled vectors plus the
reranker logit, and the loss, batch size, optimizer, early stopping, and
selection metric are the shared protocol. Evaluation uses the shared metric
code on the full split, the hard-negative set, and the action-contrast set.

    python -m src.models.transition_model.pipelines.train_token_reranker --seeds 42 43 44
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import ANNOTATION_ROOT, CHECKPOINT_ROOT, LOG_ROOT, RESULT_ROOT, VJEPA_FEATURE_ROOT
from src.eval.evaluate import save_metrics
from src.models.baselines.common import (
    build_result,
    candidate_ids,
    evaluate_full,
    evaluate_hard_negatives,
    hard_negative_video_ids,
    read_hard_negative_rows,
    split_hard_negative_rows,
)
from src.models.baselines.vjepa_baseline import ACTION_CONTRAST_FILE, VariantDataset, result_name
from src.models.transition_model.encoder import encode_text_queries
from src.models.transition_model.pipelines.train import (
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    PATIENCE,
    SELECTION_METRIC,
    WEIGHT_DECAY,
    EarlyStopping,
    checkpoint_path,
    contrastive_loss,
    log_path,
)
from src.models.transition_model.retrieval import RERANK_HIDDEN_DIM, load_checkpoint
from src.models.transition_model.transition import DROPOUT, NUM_HEADS, masked_mean
from src.models.transition_model.utils import (
    collate_transitions,
    encode_row_texts,
    feature_path,
    get_device,
    load_split_rows,
    set_random_seed,
    unique_in_order,
)


# Config

RUN_NAME = "two_stage_token_reranker"
ENCODER_VARIANT = "full_transition_without_query_conditioning"
TOKEN_VARIANT = "full_transition"  # [z, dz] tokens, as the stage-1 encoder was trained
ENCODE_BATCH_SIZE = 256
QUERY_TILE = 128
VIDEO_TILE = 256


# Model

class TokenCrossAttentionReranker(nn.Module):
    """Score (text, video) pairs from text-to-token cross-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = NUM_HEADS,
        hidden_dim: int = RERANK_HIDDEN_DIM,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(4 * dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )

    def attend(self, texts: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return query-specific video vectors (Q, V, d) from texts (Q, d), tokens (V, L, d)."""
        q_count, (v_count, length, dim) = texts.shape[0], tokens.shape
        heads, head_dim = self.num_heads, dim // self.num_heads
        q = self.query(texts).view(q_count, heads, head_dim)
        k = self.key(tokens).view(v_count, length, heads, head_dim)
        v = self.value(tokens).view(v_count, length, heads, head_dim)
        logits = torch.einsum("qhd,vlhd->qvhl", q, k) / math.sqrt(head_dim)
        logits = logits.masked_fill(~mask[None, :, None, :], float("-inf"))
        weights = self.attention_dropout(logits.softmax(dim=-1))
        attended = torch.einsum("qvhl,vlhd->qvhd", weights, v).reshape(q_count, v_count, dim)
        return F.normalize(self.norm(self.out(attended)), dim=-1)

    def forward(self, texts: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return reranker logits (Q, V)."""
        attended = self.attend(texts, tokens, mask)
        t = texts[:, None, :].expand_as(attended)
        pairs = torch.cat((t, attended, t * attended, (t - attended).abs()), dim=-1)
        return self.mlp(pairs).squeeze(-1)


class FrozenVideos:
    """Frozen pooled vectors (N, d), per-token outputs (N, L, d), and token masks (N, L)."""

    def __init__(self, pooled: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor):
        self.pooled, self.tokens, self.mask = pooled, tokens, mask

    def take(self, index) -> "FrozenVideos":
        return FrozenVideos(self.pooled[index], self.tokens[index].float(), self.mask[index])


def encode_frozen_videos(model, video_ids: list[str], feature_dir: Path) -> FrozenVideos:
    """Run the frozen stage-1 encoder; keep its pooled vector and its per-token outputs."""
    encoder = model.video_encoder
    device = next(model.parameters()).device
    dataset = VariantDataset(TOKEN_VARIANT, video_ids, list(range(len(video_ids))), feature_dir)
    loader = DataLoader(dataset, batch_size=ENCODE_BATCH_SIZE, collate_fn=collate_transitions)
    pooled, tokens, masks = [], [], []
    model.eval()
    with torch.inference_mode():
        for sequences, mask, _ in loader:
            sequences, mask = sequences.to(device), mask.to(device)
            positions = torch.arange(sequences.shape[1], device=device)
            hidden = encoder.input(sequences) + encoder.positions(positions)
            hidden = encoder.encoder(hidden, src_key_padding_mask=~mask)
            vector = F.normalize(encoder.output(masked_mean(hidden, mask)), dim=-1)
            if not pooled:
                # The pooled vector must be exactly what the stage-1 model scores with.
                expected = model.encode_videos(sequences, mask)
                torch.testing.assert_close(vector, expected, rtol=1e-4, atol=1e-4)
            pooled.append(vector)
            tokens.append(encoder.output(hidden).half())
            masks.append(mask)
    length = max(t.shape[1] for t in tokens)
    tokens = torch.cat([F.pad(t, (0, 0, 0, length - t.shape[1])) for t in tokens])
    masks = torch.cat([F.pad(m, (0, length - m.shape[1])) for m in masks])
    return FrozenVideos(torch.cat(pooled).clone(), tokens.clone(), masks.clone())


def pair_scores(model, reranker, texts: torch.Tensor, videos: FrozenVideos) -> torch.Tensor:
    """Frozen scaled cosine of pooled vectors plus the token reranker logit, (Q, V)."""
    cosine = model.logit_scale.exp().clamp(max=100) * texts @ videos.pooled.T
    return cosine + reranker(texts, videos.tokens.float(), videos.mask)


def score_all(model, reranker, texts: torch.Tensor, videos: FrozenVideos) -> np.ndarray:
    """Score encoded texts (Q, d) against all videos in tiles; return a (Q, V) array."""
    reranker.eval()
    out = np.empty((len(texts), len(videos.pooled)), dtype=np.float32)
    with torch.inference_mode():
        for q in range(0, len(texts), QUERY_TILE):
            for c in range(0, len(videos.pooled), VIDEO_TILE):
                tile = pair_scores(
                    model, reranker, texts[q : q + QUERY_TILE], videos.take(slice(c, c + VIDEO_TILE))
                )
                out[q : q + QUERY_TILE, c : c + VIDEO_TILE] = tile.float().cpu().numpy()
    return out


def encode_texts(model, raw: np.ndarray) -> torch.Tensor:
    device = next(model.parameters()).device
    with torch.inference_mode():
        return model.encode_texts(torch.from_numpy(np.asarray(raw, dtype=np.float32)).to(device)).clone()


def evaluate_candidates(model, reranker, path, rows, raw_texts, feature_dir):
    """Score one hard-negative-schema file with the shared evaluator; (None, 0) if missing."""
    candidate_rows = read_hard_negative_rows(path, rows)
    available = {
        v for v in hard_negative_video_ids(candidate_rows) if feature_path(feature_dir, v).exists()
    }
    kept, skipped = split_hard_negative_rows(candidate_rows, available)
    if not kept:
        return None, skipped
    video_ids = hard_negative_video_ids(kept)
    videos = encode_frozen_videos(model, video_ids, feature_dir)
    reranker.eval()

    def score_fn(text, index):
        with torch.inference_mode():
            index = torch.as_tensor(index, device=videos.pooled.device)
            scores = pair_scores(model, reranker, encode_texts(model, text), videos.take(index))
        return scores.float().cpu().numpy()

    # evaluate_hard_negatives indexes the "embeddings" by candidate; pass row indices.
    positions = np.arange(len(video_ids))
    metrics = evaluate_hard_negatives(kept, rows, raw_texts, video_ids, positions, score_fn)
    return metrics, skipped


# Main functions

def train_token_reranker(
    seed: int,
    *,
    feature_dir: Path = VJEPA_FEATURE_ROOT,
    checkpoint_dir: Path = CHECKPOINT_ROOT,
    log_dir: Path = LOG_ROOT,
    epochs: int = NUM_EPOCHS,
    patience: int | None = PATIENCE,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    device: str | None = None,
    limit: int | None = None,
):
    """Train reranker B on the frozen stage-1 encoder; return (model, reranker, checkpoint dict)."""
    device = get_device(device)
    encoder_checkpoint = checkpoint_path(checkpoint_dir, result_name(ENCODER_VARIANT), seed)
    model, stage1 = load_checkpoint(encoder_checkpoint, device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    text_encoder, text_model_name = stage1["text_encoder"], stage1["text_model_name"]

    set_random_seed(seed)
    reranker = TokenCrossAttentionReranker(model.config["model_dim"]).to(device)

    train_rows, _ = load_split_rows(ANNOTATION_ROOT / "train.jsonl", feature_dir, limit)
    val_rows, _ = load_split_rows(ANNOTATION_ROOT / "val.jsonl", feature_dir, limit)
    print(f"[{RUN_NAME}] encoding {len(train_rows)} train rows with the frozen encoder", flush=True)
    train_ids = [str(row["target_video_id"]) for row in train_rows]
    unique_train = unique_in_order(train_ids)
    train_videos = encode_frozen_videos(model, unique_train, feature_dir)
    position = {video_id: i for i, video_id in enumerate(unique_train)}
    row_video = torch.tensor([position[v] for v in train_ids], device=device)
    texts = unique_in_order(row["text_query"] for row in train_rows)
    text_index = {text: i for i, text in enumerate(texts)}
    text_bank = encode_texts(
        model,
        encode_text_queries(texts, encoder_name=text_encoder, model_name=text_model_name, device=device),
    )
    row_text = torch.tensor([text_index[row["text_query"]] for row in train_rows], device=device)

    val_texts = encode_texts(model, encode_row_texts(val_rows, text_encoder, text_model_name, device))
    val_candidates = candidate_ids(val_rows)
    val_videos = encode_frozen_videos(model, val_candidates, feature_dir)

    def validate():
        return evaluate_full(score_all(model, reranker, val_texts, val_videos), val_rows, val_candidates)

    optimizer = torch.optim.AdamW(reranker.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    stopper = EarlyStopping(patience)
    best_path = checkpoint_path(checkpoint_dir, RUN_NAME, seed)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    train_log = log_path(log_dir, RUN_NAME, seed)
    train_log.parent.mkdir(parents=True, exist_ok=True)
    train_log.write_text("", encoding="utf-8")
    num_batches = max(len(train_rows) // batch_size, 1)
    metadata = {
        "variant": RUN_NAME,
        "seed": seed,
        "encoder_checkpoint": str(encoder_checkpoint),
        "text_encoder": text_encoder,
        "text_model_name": text_model_name,
        "reranker_config": {"dim": model.config["model_dim"], "num_heads": reranker.num_heads},
    }

    for epoch in range(1, epochs + 1):
        reranker.train()
        order = torch.randperm(len(train_rows), generator=generator).to(device)
        total = 0.0
        for b in range(num_batches):
            index = order[b * batch_size : (b + 1) * batch_size]
            ids = row_text[index]
            scores = pair_scores(model, reranker, text_bank[ids], train_videos.take(row_video[index]))
            loss = contrastive_loss(scores, ids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        metrics = validate()
        loss = total / num_batches
        with train_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"epoch": epoch, "train_loss": loss, "val": metrics}) + "\n")
        print(
            f"[{RUN_NAME}] Epoch {epoch}: loss={loss:.4f} "
            f"val R@1={metrics['recall@1']:.4f} MRR={metrics['mrr']:.4f}",
            flush=True,
        )
        if stopper.update(epoch, metrics[SELECTION_METRIC]):
            torch.save(
                {"reranker_state_dict": reranker.state_dict(), "epoch": epoch, "val_metrics": metrics,
                 "selection_metric": SELECTION_METRIC, **metadata},
                best_path,
            )
        if stopper.should_stop:
            break

    checkpoint = torch.load(best_path, map_location=device)
    checkpoint["training_summary"] = stopper.summary(epochs)
    torch.save(checkpoint, best_path)
    reranker.load_state_dict(checkpoint["reranker_state_dict"])
    print(f"[{RUN_NAME}] best epoch {stopper.best_epoch} of {stopper.last_epoch}", flush=True)
    return model, reranker, checkpoint


def evaluate_token_reranker(model, reranker, checkpoint, output: Path, feature_dir: Path, device, limit=None):
    """Evaluate on test: full split, hard negatives, and action contrast."""
    rows, skipped = load_split_rows(ANNOTATION_ROOT / "test.jsonl", feature_dir, limit)
    raw_texts = encode_row_texts(rows, checkpoint["text_encoder"], checkpoint["text_model_name"], device)
    candidates = candidate_ids(rows)
    videos = encode_frozen_videos(model, candidates, feature_dir)
    scores = score_all(model, reranker, encode_texts(model, raw_texts), videos)
    summary = checkpoint["training_summary"]
    result = build_result(
        RUN_NAME,
        rows,
        evaluate_full(scores, rows, candidates),
        variant=RUN_NAME,
        checkpoint_path=str(checkpoint_path(CHECKPOINT_ROOT, RUN_NAME, checkpoint["seed"])),
        encoder_checkpoint=checkpoint["encoder_checkpoint"],
        seed=checkpoint["seed"],
        best_epoch=summary["best_epoch"],
        epochs_run=summary["epochs_run"],
        training_summary=summary,
        video_input="state_transition",
        query_conditioned=True,
        reranker="token_cross_attention",
        skipped_missing_features=skipped,
    )
    for key, path in (
        ("hard_negative", ANNOTATION_ROOT / "hard_negatives.jsonl"),
        ("action_contrast", ANNOTATION_ROOT / ACTION_CONTRAST_FILE),
    ):
        metrics, set_skipped = evaluate_candidates(model, reranker, path, rows, raw_texts, feature_dir)
        result[key] = metrics
        result[f"{key}_skipped"] = set_skipped
    save_metrics(result, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--feature-dir", type=Path, default=VJEPA_FEATURE_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--result-dir", type=Path, default=RESULT_ROOT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    for seed in args.seeds:
        model, reranker, checkpoint = train_token_reranker(
            seed, feature_dir=args.feature_dir, checkpoint_dir=args.checkpoint_dir,
            device=device, limit=args.limit,
        )
        output = args.result_dir / f"{RUN_NAME}_seed{seed}.json"
        result = evaluate_token_reranker(model, reranker, checkpoint, output, args.feature_dir, device, args.limit)
        print(json.dumps({f"{RUN_NAME}_seed{seed}": result["full"]}, indent=2))
        print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
