"""Two-stage training of the query-conditioned token-level matcher (final method).

Stage 1 trains the temporal encoder with the cosine objective only (no
reranker), with the shared protocol of pipelines.train. Stage 2 freezes it and
trains the matcher: the text attends over the encoder's per-token outputs
(multi-head cross-attention), which gives a query-specific video vector a, and
an MLP scores [t, a, t*a, |t-a|]. The final score is the frozen scaled cosine of
the pooled vectors plus the matcher logit. The loss, batch size, optimizer,
early stopping, and selection metric are the shared protocol in config.py.

--encoder picks the video tokens; everything else is identical:

- full_transition (E2, proposed): tokens [z[t], dz[t]] (video_input
  state_transition), stage-1 checkpoint
  ablation_full_transition_without_query_conditioning_seed<seed>; run name
  two_stage_token_reranker.
- state_only (E1, matched control): tokens [z[t], 0] for t = 1..T (video_input
  state_only), same token width and parameter count, and the same states E2 can
  recover; stage-1 checkpoint ablation_state_only_without_query_conditioning_seed<seed>,
  trained with
  python -m src.models.transition_model.pipelines.train --run-name ablation_state_only_without_query_conditioning --video-input state_only --no-query-conditioning --seed <seed>;
  run name two_stage_token_reranker_state_only.

Evaluation uses the shared metric code on the full split, the hard-negative
set, and the action-contrast set, and writes per-query ranks
(per_query_<run>_seed<seed>.jsonl: query_id and the target rank in each
setting, for paired comparisons) and rankings (rankings_<run>_seed<seed>.jsonl)
for the error analysis.

    python -m src.models.transition_model.pipelines.train_token_reranker --encoder full_transition --seeds 42 43 44
    python -m src.models.transition_model.pipelines.train_token_reranker --encoder state_only --seeds 42 43 44
    # re-evaluate trained checkpoints only (per-query ranks; --reverse-time for reversed states):
    python -m src.models.transition_model.pipelines.train_token_reranker --encoder full_transition --eval-only --seeds 42 43 44
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
from src.eval.candidate_metrics import summarize_candidate_set
from src.eval.evaluate import save_metrics
from src.models.baselines.common import (
    build_result,
    candidate_ids,
    evaluate_full,
    hard_negative_video_ids,
    query_ids,
    ranking_records,
    read_hard_negative_rows,
    save_jsonl,
    split_hard_negative_rows,
    target_ids,
)
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
    TransitionDataset,
    collate_transitions,
    encode_row_texts,
    feature_path,
    get_device,
    load_split_rows,
    set_random_seed,
    unique_in_order,
)


# Config

# --encoder -> (run name, stage-1 run name, stage-1 video_input)
ENCODERS = {
    "full_transition": (
        "two_stage_token_reranker",
        "ablation_full_transition_without_query_conditioning",
        "state_transition",
    ),
    "state_only": (
        "two_stage_token_reranker_state_only",
        "ablation_state_only_without_query_conditioning",
        "state_only",
    ),
}
ACTION_CONTRAST_FILE = "action_contrast.jsonl"
DEFAULT_ENCODER = "full_transition"
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


def encode_frozen_videos(
    model, video_ids: list[str], feature_dir: Path, reverse_time: bool = False
) -> FrozenVideos:
    """Run the frozen stage-1 encoder; keep its pooled vector and its per-token outputs.

    Tokens use the stage-1 model's own video_input, so they match its training.
    """
    encoder = model.video_encoder
    device = next(model.parameters()).device
    dataset = TransitionDataset(
        video_ids, list(range(len(video_ids))), feature_dir, model.video_input, reverse_time
    )
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


def target_rank(scores: np.ndarray, candidates: list[str], target: str) -> int:
    """1-based rank of target, with the stable tie rule of the shared evaluator."""
    order = np.argsort(-np.asarray(scores, dtype=np.float32), kind="stable")
    return int(np.where(order == candidates.index(target))[0][0]) + 1


def evaluate_candidates(
    model, reranker, path, rows, raw_texts, feature_dir, reverse_time=False, ranks_out=None
):
    """Score one hard-negative-schema file with the shared evaluator; (None, 0) if missing.

    ranks_out, if given, is filled with query_id -> target rank in that set.
    """
    candidate_rows = read_hard_negative_rows(path, rows)
    available = {
        v for v in hard_negative_video_ids(candidate_rows) if feature_path(feature_dir, v).exists()
    }
    kept, skipped = split_hard_negative_rows(candidate_rows, available)
    if not kept:
        return None, skipped
    video_ids = hard_negative_video_ids(kept)
    videos = encode_frozen_videos(model, video_ids, feature_dir, reverse_time)
    position = {v: i for i, v in enumerate(video_ids)}
    text_by_query = dict(zip(query_ids(rows), raw_texts))
    reranker.eval()
    lookup = {}
    for row in kept:
        query_id = str(row["query_id"])
        candidates = [str(v) for v in row["candidate_video_ids"]]
        index = torch.as_tensor([position[v] for v in candidates], device=videos.pooled.device)
        with torch.inference_mode():
            text = encode_texts(model, text_by_query[query_id][None])
            scores = pair_scores(model, reranker, text, videos.take(index))[0].float().cpu().numpy()
        lookup.update({(query_id, v): float(x) for v, x in zip(candidates, scores)})
        if ranks_out is not None:
            ranks_out[query_id] = target_rank(scores, candidates, str(row["target_video_id"]))
    return summarize_candidate_set(kept, lookup), skipped


def per_query_records(rows, scores, candidates, set_ranks) -> list[dict]:
    """One record per test query: query_id and the target rank in every setting (None if absent)."""
    return [
        {
            "query_id": query_id,
            "full_rank": target_rank(row_scores, candidates, target),
            **{f"{name}_rank": ranks.get(query_id) for name, ranks in set_ranks.items()},
        }
        for query_id, target, row_scores in zip(query_ids(rows), target_ids(rows), scores)
    ]


def save_result(result: dict, output: Path, overwrite: bool = True) -> None:
    """Save result; with overwrite=False keep an existing file and report whether it reproduces."""
    output = Path(output)
    if overwrite or not output.exists():
        save_metrics(result, output)
        return
    saved = json.loads(output.read_text(encoding="utf-8"))
    diffs = [
        abs(result[key][metric] - saved[key][metric])
        for key in ("full", "hard_negative", "action_contrast")
        if result.get(key) and saved.get(key)
        for metric in ("recall@1", "recall@5", "recall@10", "mrr")
    ]
    status = "reproduced" if diffs and max(diffs) < 1e-4 else "DIFFERS from"
    print(f"kept {output}; re-evaluation {status} it (max |diff| = {max(diffs, default=0):.2e})")


# Main functions

def load_stage1(encoder: str, seed: int, checkpoint_dir: Path, device: str):
    """Return the frozen stage-1 model, its checkpoint dict, and its path."""
    _, stage1_run, video_input = ENCODERS[encoder]
    path = checkpoint_path(checkpoint_dir, stage1_run, seed)
    if not path.exists():
        raise FileNotFoundError(f"missing stage-1 checkpoint {path}")
    model, stage1 = load_checkpoint(path, device)
    if model.video_input != video_input or model.reranker is not None:
        raise ValueError(
            f"{path} must be a cosine-only encoder with video_input {video_input!r}, "
            f"got {model.video_input!r} (reranker: {model.reranker is not None})"
        )
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, stage1, path


def load_token_reranker(encoder: str, seed: int, checkpoint_dir: Path, device: str):
    """Rebuild a trained matcher and its frozen stage-1 model; return (model, reranker, checkpoint)."""
    run_name = ENCODERS[encoder][0]
    model, _, _ = load_stage1(encoder, seed, checkpoint_dir, device)
    checkpoint = torch.load(checkpoint_path(checkpoint_dir, run_name, seed), map_location=device)
    config = checkpoint["reranker_config"]
    reranker = TokenCrossAttentionReranker(config["dim"], config["num_heads"]).to(device)
    reranker.load_state_dict(checkpoint["reranker_state_dict"])
    return model, reranker.eval(), checkpoint


def train_token_reranker(
    seed: int,
    *,
    encoder: str = DEFAULT_ENCODER,
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
    """Train the token-level matcher on the frozen stage-1 encoder.

    Returns (model, reranker, checkpoint dict).
    """
    device = get_device(device)
    run_name, _, video_input = ENCODERS[encoder]
    model, stage1, encoder_checkpoint = load_stage1(encoder, seed, checkpoint_dir, device)
    text_encoder, text_model_name = stage1["text_encoder"], stage1["text_model_name"]

    set_random_seed(seed)
    reranker = TokenCrossAttentionReranker(model.config["model_dim"]).to(device)

    train_rows, _ = load_split_rows(ANNOTATION_ROOT / "train.jsonl", feature_dir, limit)
    val_rows, _ = load_split_rows(ANNOTATION_ROOT / "val.jsonl", feature_dir, limit)
    print(f"[{run_name}] encoding {len(train_rows)} train rows with the frozen encoder", flush=True)
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
    best_path = checkpoint_path(checkpoint_dir, run_name, seed)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    train_log = log_path(log_dir, run_name, seed)
    train_log.parent.mkdir(parents=True, exist_ok=True)
    train_log.write_text("", encoding="utf-8")
    num_batches = max(len(train_rows) // batch_size, 1)
    metadata = {
        "variant": run_name,
        "encoder": encoder,
        "video_input": video_input,
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
            f"[{run_name}] Epoch {epoch}: loss={loss:.4f} "
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
    print(f"[{run_name}] best epoch {stopper.best_epoch} of {stopper.last_epoch}", flush=True)
    return model, reranker, checkpoint


def evaluate_token_reranker(
    model,
    reranker,
    checkpoint,
    output: Path,
    feature_dir: Path,
    device,
    limit=None,
    *,
    encoder: str = DEFAULT_ENCODER,
    checkpoint_dir: Path = CHECKPOINT_ROOT,
    per_query_path: Path | None = None,
    rankings_path: Path | None = None,
    reverse_time: bool = False,
    overwrite: bool = True,
):
    """Evaluate on test: full split, hard negatives, and action contrast."""
    run_name, _, video_input = ENCODERS[encoder]
    rows, skipped = load_split_rows(ANNOTATION_ROOT / "test.jsonl", feature_dir, limit)
    raw_texts = encode_row_texts(rows, checkpoint["text_encoder"], checkpoint["text_model_name"], device)
    candidates = candidate_ids(rows)
    videos = encode_frozen_videos(model, candidates, feature_dir, reverse_time)
    scores = score_all(model, reranker, encode_texts(model, raw_texts), videos)
    summary = checkpoint["training_summary"]
    result = build_result(
        run_name,
        rows,
        evaluate_full(scores, rows, candidates),
        variant=run_name,
        checkpoint_path=str(checkpoint_path(checkpoint_dir, run_name, checkpoint["seed"])),
        encoder_checkpoint=checkpoint["encoder_checkpoint"],
        seed=checkpoint["seed"],
        best_epoch=summary["best_epoch"],
        epochs_run=summary["epochs_run"],
        training_summary=summary,
        video_input=video_input,
        query_conditioned=True,
        reranker="token_cross_attention",
        reverse_time=reverse_time,
        skipped_missing_features=skipped,
    )
    set_ranks = {}
    for key, path in (
        ("hard_negative", ANNOTATION_ROOT / "hard_negatives.jsonl"),
        ("action_contrast", ANNOTATION_ROOT / ACTION_CONTRAST_FILE),
    ):
        set_ranks[key] = {}
        metrics, set_skipped = evaluate_candidates(
            model, reranker, path, rows, raw_texts, feature_dir, reverse_time,
            set_ranks[key],
        )
        result[key] = metrics
        result[f"{key}_skipped"] = set_skipped
    save_result(result, output, overwrite)
    if per_query_path is not None:
        save_jsonl(per_query_records(rows, scores, candidates, set_ranks), per_query_path)
    if rankings_path is not None:
        save_jsonl(ranking_records(run_name, rows, scores, candidates), rankings_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=tuple(ENCODERS), default=DEFAULT_ENCODER)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--feature-dir", type=Path, default=VJEPA_FEATURE_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--result-dir", type=Path, default=RESULT_ROOT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="re-evaluate trained checkpoints (no training); existing result files are kept",
    )
    parser.add_argument(
        "--reverse-time",
        action="store_true",
        help="evaluate trained checkpoints with states reversed in time (implies --eval-only)",
    )
    args = parser.parse_args()

    device = get_device(args.device)
    run_name = ENCODERS[args.encoder][0]
    eval_only = args.eval_only or args.reverse_time
    for seed in args.seeds:
        if eval_only:
            model, reranker, checkpoint = load_token_reranker(
                args.encoder, seed, args.checkpoint_dir, device
            )
        else:
            model, reranker, checkpoint = train_token_reranker(
                seed, encoder=args.encoder, feature_dir=args.feature_dir,
                checkpoint_dir=args.checkpoint_dir, device=device, limit=args.limit,
            )
        stem = f"{run_name}_seed{seed}" + ("_reversed" if args.reverse_time else "")
        output = args.result_dir / f"{stem}.json"
        result = evaluate_token_reranker(
            model, reranker, checkpoint, output, args.feature_dir, device, args.limit,
            encoder=args.encoder, checkpoint_dir=args.checkpoint_dir,
            per_query_path=args.result_dir / f"per_query_{stem}.jsonl",
            rankings_path=args.result_dir / f"rankings_{stem}.jsonl",
            reverse_time=args.reverse_time, overwrite=not eval_only,
        )
        print(json.dumps({stem: result["full"]}, indent=2))
        print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
