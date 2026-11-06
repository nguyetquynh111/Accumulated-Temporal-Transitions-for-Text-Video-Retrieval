"""B2 baseline (pooled V-JEPA, no transitions) and the transition ablations (T3.3).

Every variant feeds a different view of the same cached V-JEPA states into the
shared TransitionRetrievalModel, trains it with the main contrastive loss, and
is scored with the shared evaluator. Only the video input, and for two variants
the query-conditioned reranker, changes between runs:

- pooled: mean state as one token, no reranker (B2).
- full_transition: the z[t + 1] - z[t] sequence with the reranker. This is the
  main model; evaluate Quynh's checkpoint with it to get comparable rankings.
- no_transition: the raw state sequence z[1:T] instead of transitions.
- mean_transition: the mean transition vector as one token.
- final_minus_initial: z[T] - z[1] as one token.
- full_transition_without_query_conditioning: transitions, no reranker (B3).

Note that the mean of z[t + 1] - z[t] telescopes to (z[T] - z[1]) / (T - 1), and
the model's input LayerNorm removes that scale, so mean_transition and
final_minus_initial see the same input up to a constant factor.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import (
    ANNOTATION_ROOT,
    CHECKPOINT_ROOT,
    FEATURE_ROOT,
    RANDOM_SEED,
    RESULT_ROOT,
    TEXT_MODEL_NAME,
)
from src.eval.evaluate import save_metrics
from src.models.baselines.common import (
    build_result,
    candidate_ids,
    evaluate_full,
    evaluate_hard_negatives,
    hard_negative_video_ids,
    ranking_records,
    read_hard_negative_rows,
    save_jsonl,
    split_hard_negative_rows,
)
from src.models.transition_model.encoder import encode_text_queries
from src.models.transition_model.pipelines.evaluate import encode_row_texts
from src.models.transition_model.pipelines.train import train_one_epoch
from src.models.transition_model.retrieval import (
    TransitionRetrievalModel,
    load_checkpoint,
    save_checkpoint,
    score_transition_retrieval,
)
from src.models.transition_model.transition import compute_transitions
from src.models.transition_model.utils import (
    collate_transitions,
    ensure_dir,
    feature_path,
    get_device,
    load_split_rows,
    load_states,
    pad_sequences,
    set_random_seed,
)


# Config

# Variant name -> whether the query-conditioned reranker is used.
VARIANTS = {
    "pooled": False,
    "full_transition": True,
    "no_transition": True,
    "mean_transition": True,
    "final_minus_initial": True,
    "full_transition_without_query_conditioning": False,
}
ABLATION_VARIANTS = (
    "no_transition",
    "mean_transition",
    "final_minus_initial",
    "full_transition_without_query_conditioning",
)
MAIN_CHECKPOINT = CHECKPOINT_ROOT / "transition_model_best.pt"
BATCH_SIZE = 128
ENCODE_BATCH_SIZE = 256
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
NUM_WORKERS = 0
SELECTION_METRIC = "mrr"
TEXT_ENCODER = "minilm"


# Sub functions

def result_name(variant: str) -> str:
    """Return the model name used in result, ranking, and checkpoint files."""
    if variant == "pooled":
        return "vjepa_baseline"
    if variant == "full_transition":
        return "transition_model"
    return f"ablation_{variant}"


def variant_sequence(
    variant: str,
    state_embeddings: np.ndarray,
    transition_embeddings: np.ndarray | None = None,
) -> np.ndarray:
    """Return the (L, D) video input one variant feeds to the retrieval model."""
    states = np.asarray(state_embeddings, dtype=np.float32)
    if states.ndim != 2 or states.shape[0] < 2:
        raise ValueError("state_embeddings must have shape (T, D) with T >= 2")
    if variant == "pooled":
        return states.mean(axis=0, keepdims=True)
    if variant == "no_transition":
        return states
    if variant == "final_minus_initial":
        return (states[-1] - states[0])[None]
    transitions = (
        compute_transitions(states)
        if transition_embeddings is None
        else np.asarray(transition_embeddings, dtype=np.float32)
    )
    if variant in ("full_transition", "full_transition_without_query_conditioning"):
        return transitions
    if variant == "mean_transition":
        return transitions.mean(axis=0, keepdims=True)
    raise ValueError(f"unknown variant: {variant}")


class VariantDataset(Dataset):
    """Yield (variant sequence, label) pairs, reading feature caches lazily."""

    def __init__(self, variant: str, video_ids: list[str], labels: list[int], feature_dir: Path):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant: {variant}")
        if len(video_ids) != len(labels):
            raise ValueError("video_ids and labels must have the same length")
        self.variant = variant
        self.video_ids = video_ids
        self.labels = labels
        self.feature_dir = Path(feature_dir)

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        states = load_states(self.feature_dir, self.video_ids[index])
        return variant_sequence(self.variant, states), self.labels[index]


def encode_variant_bank(
    model: TransitionRetrievalModel,
    variant: str,
    video_ids: list[str],
    feature_dir: Path,
    batch_size: int = ENCODE_BATCH_SIZE,
) -> np.ndarray:
    """Return encoded video representations (N, model_dim) in video_ids order."""
    dataset = VariantDataset(variant, video_ids, list(range(len(video_ids))), feature_dir)
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=NUM_WORKERS, collate_fn=collate_transitions
    )
    model.eval()
    device = next(model.parameters()).device
    chunks = []
    with torch.inference_mode():
        for sequences, mask, _ in loader:
            videos = model.encode_videos(sequences.to(device), mask.to(device))
            chunks.append(videos.float().cpu().numpy())
    return np.concatenate(chunks)


def score_variant_rows(
    model: TransitionRetrievalModel,
    variant: str,
    rows: list[dict],
    text_embeddings: np.ndarray,
    feature_dir: Path,
) -> tuple[np.ndarray, list[str], dict]:
    """Score every query against the split's target videos; return scores, candidates, metrics."""
    candidates = candidate_ids(rows)
    videos = encode_variant_bank(model, variant, candidates, feature_dir)
    scores = score_transition_retrieval(text_embeddings, videos, model)
    return scores, candidates, evaluate_full(scores, rows, candidates)


def build_variant_loader(
    variant: str,
    rows: list[dict],
    feature_dir: Path,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, list[str]]:
    """Return a shuffled loader of (variant sequence, text id) and the unique texts."""
    texts = list(dict.fromkeys(row["text_query"] for row in rows))
    text_index = {text: i for i, text in enumerate(texts)}
    dataset = VariantDataset(
        variant,
        [str(row["target_video_id"]) for row in rows],
        [text_index[row["text_query"]] for row in rows],
        feature_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(dataset) > batch_size,
        num_workers=NUM_WORKERS,
        collate_fn=collate_transitions,
        generator=torch.Generator().manual_seed(seed),
    )
    return loader, texts


# Main functions

def train_variant_model(
    variant: str,
    train_annotations: Path,
    val_annotations: Path,
    feature_dir: Path,
    checkpoint_dir: Path,
    *,
    epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    text_encoder: str = TEXT_ENCODER,
    text_model_name: str = TEXT_MODEL_NAME,
    device: str | None = None,
    limit: int | None = None,
    seed: int = RANDOM_SEED,
) -> Path:
    """Train one baseline/ablation variant and return its best checkpoint path."""
    if variant == "full_transition":
        raise ValueError(
            "train the main model with src.models.transition_model.pipelines.train"
        )
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    set_random_seed(seed)
    device = get_device(device)
    checkpoint_dir = ensure_dir(checkpoint_dir)
    name = result_name(variant)
    best_path = checkpoint_dir / f"{name}_best.pt"
    log_path = checkpoint_dir / f"{name}_train_log.jsonl"

    train_rows, train_skipped = load_split_rows(train_annotations, feature_dir, limit)
    val_rows, val_skipped = load_split_rows(val_annotations, feature_dir, limit)
    if not train_rows or not val_rows:
        raise ValueError(f"train and val need rows with cached features in {feature_dir}")
    print(f"[{name}] train rows: {len(train_rows)} (skipped {train_skipped} without features)")
    print(f"[{name}] val rows: {len(val_rows)} (skipped {val_skipped} without features)")

    loader, train_texts = build_variant_loader(variant, train_rows, feature_dir, batch_size, seed)
    text_bank = torch.from_numpy(
        encode_text_queries(
            train_texts, encoder_name=text_encoder, model_name=text_model_name, device=device
        )
    ).to(device)
    val_texts = encode_row_texts(val_rows, text_encoder, text_model_name, device)

    video_dim = load_states(feature_dir, str(train_rows[0]["target_video_id"])).shape[1]
    model = TransitionRetrievalModel(
        video_dim, text_bank.shape[1], query_conditioned=VARIANTS[variant]
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)

    best_score = float("-inf")
    log_path.write_text("", encoding="utf-8")
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, loader, text_bank, optimizer)
        _, _, metrics = score_variant_rows(model, variant, val_rows, val_texts, feature_dir)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"epoch": epoch, "train_loss": loss, "val": metrics}) + "\n")
        print(
            f"[{name}] epoch {epoch}: loss={loss:.4f} "
            f"val R@1={metrics['recall@1']:.4f} MRR={metrics['mrr']:.4f}"
        )
        if metrics[SELECTION_METRIC] > best_score:
            best_score = metrics[SELECTION_METRIC]
            save_checkpoint(
                model,
                best_path,
                variant=variant,
                epoch=epoch,
                val_metrics=metrics,
                selection_metric=SELECTION_METRIC,
                text_encoder=text_encoder,
                text_model_name=text_model_name,
                feature_dir=str(feature_dir),
                seed=seed,
            )
    return best_path


def score_vjepa_baseline(
    text_embeddings: np.ndarray,
    video_embeddings: np.ndarray,
    model: TransitionRetrievalModel | None = None,
) -> np.ndarray:
    """Return score matrix with shape (N_queries, N_candidates).

    Without a model, scores are dot products of same-sized embeddings. V-JEPA
    and text embeddings are not aligned, so real runs pass the trained pooled
    model and raw pooled V-JEPA embeddings (N_candidates, D).
    """
    text_embeddings = np.asarray(text_embeddings, dtype=np.float32)
    video_embeddings = np.asarray(video_embeddings, dtype=np.float32)
    if text_embeddings.ndim != 2 or video_embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D arrays")
    if model is None:
        if text_embeddings.shape[1] != video_embeddings.shape[1]:
            raise ValueError("text and video embedding dimensions must match")
        return (text_embeddings @ video_embeddings.T).astype(np.float32)
    return encode_and_score(model, text_embeddings, [v[None] for v in video_embeddings])


def encode_and_score(
    model: TransitionRetrievalModel,
    text_embeddings: np.ndarray,
    sequences: list[np.ndarray],
) -> np.ndarray:
    """Encode variable-length video sequences with model and score texts against them."""
    padded, mask = pad_sequences(sequences)
    model.eval()
    device = next(model.parameters()).device
    with torch.inference_mode():
        videos = model.encode_videos(
            torch.from_numpy(padded).to(device), torch.from_numpy(mask).to(device)
        )
    return score_transition_retrieval(text_embeddings, videos.float().cpu().numpy(), model)


def run_transition_ablation(
    variant: str,
    text_embeddings: np.ndarray,
    state_embeddings: dict[str, np.ndarray],
    transition_embeddings: dict[str, np.ndarray],
    model: TransitionRetrievalModel | None = None,
) -> np.ndarray:
    """Return score matrix for one ablation variant.

    Candidates are the keys of state_embeddings, in order. Without a model,
    each video input is mean-pooled and dot-scored, which needs matching text
    and video dimensions; with the variant's trained model, text_embeddings are
    raw text-encoder outputs.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    sequences = [
        variant_sequence(variant, states, transition_embeddings.get(video_id))
        for video_id, states in state_embeddings.items()
    ]
    if model is None:
        videos = np.stack([sequence.mean(axis=0) for sequence in sequences])
        return score_transition_retrieval(text_embeddings, videos)
    return encode_and_score(model, text_embeddings, sequences)


def evaluate_variant(
    variant: str,
    checkpoint_path: Path,
    annotations: Path,
    feature_dir: Path,
    output_path: Path,
    *,
    hard_negatives: Path | None = None,
    rankings_path: Path | None = None,
    device: str | None = None,
    limit: int | None = None,
) -> dict:
    """Evaluate one variant checkpoint on full and hard-negative retrieval and save results."""
    device = get_device(device)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    if checkpoint["model_config"]["query_conditioned"] != VARIANTS[variant]:
        raise ValueError(f"checkpoint reranker setting does not match variant {variant}")
    rows, skipped = load_split_rows(annotations, feature_dir, limit)
    if not rows:
        raise ValueError(f"no annotation rows with cached features in {feature_dir}")
    texts = encode_row_texts(rows, checkpoint["text_encoder"], checkpoint["text_model_name"], device)
    scores, candidates, full = score_variant_rows(model, variant, rows, texts, feature_dir)

    name = result_name(variant)
    result = build_result(
        name,
        rows,
        full,
        variant=variant,
        checkpoint_path=str(checkpoint_path),
        query_conditioned=VARIANTS[variant],
        skipped_missing_features=skipped,
    )
    hard_rows = read_hard_negative_rows(hard_negatives, rows)
    available = {
        v for v in hard_negative_video_ids(hard_rows) if feature_path(feature_dir, v).exists()
    }
    kept, hard_skipped = split_hard_negative_rows(hard_rows, available)
    if kept:
        video_ids = hard_negative_video_ids(kept)
        videos = encode_variant_bank(model, variant, video_ids, feature_dir)
        result["hard_negative"] = evaluate_hard_negatives(
            kept,
            rows,
            texts,
            video_ids,
            videos,
            lambda text, candidates: score_transition_retrieval(text, candidates, model),
        )
        result["hard_negative_skipped"] = hard_skipped

    save_metrics(result, output_path)
    if rankings_path is not None:
        save_jsonl(ranking_records(name, rows, scores, candidates), rankings_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        nargs="+",
        choices=(*VARIANTS, "all_ablations"),
        default=["pooled"],
        help="one or more variants; all_ablations runs the four T3.3 variants",
    )
    parser.add_argument("--train-annotations", type=Path, default=ANNOTATION_ROOT / "train.jsonl")
    parser.add_argument("--val-annotations", type=Path, default=ANNOTATION_ROOT / "val.jsonl")
    parser.add_argument("--test-annotations", type=Path, default=ANNOTATION_ROOT / "test.jsonl")
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_ROOT / "vjepa")
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="evaluate this checkpoint instead of training (one variant only)",
    )
    parser.add_argument(
        "--hard-negatives", type=Path, default=ANNOTATION_ROOT / "hard_negatives.jsonl"
    )
    parser.add_argument("--result-dir", type=Path, default=RESULT_ROOT)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--text-encoder", choices=("minilm", "mock"), default=TEXT_ENCODER)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    variants = []
    for variant in args.variant:
        variants.extend(ABLATION_VARIANTS if variant == "all_ablations" else [variant])
    variants = list(dict.fromkeys(variants))
    if args.checkpoint is not None and len(variants) != 1:
        parser.error("--checkpoint needs exactly one variant")

    for variant in variants:
        checkpoint = args.checkpoint
        if checkpoint is None and variant == "full_transition":
            checkpoint = MAIN_CHECKPOINT
        if checkpoint is None:
            checkpoint = train_variant_model(
                variant,
                args.train_annotations,
                args.val_annotations,
                args.feature_dir,
                args.checkpoint_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                text_encoder=args.text_encoder,
                device=args.device,
                limit=args.limit,
                seed=args.seed,
            )
        name = result_name(variant)
        output = args.result_dir / f"{name}.json"
        result = evaluate_variant(
            variant,
            checkpoint,
            args.test_annotations,
            args.feature_dir,
            output,
            hard_negatives=args.hard_negatives,
            rankings_path=args.result_dir / f"rankings_{name}.jsonl",
            device=args.device,
            limit=args.limit,
        )
        print(json.dumps({name: {"full": result["full"], "hard_negative": result["hard_negative"]}}, indent=2))
        print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
