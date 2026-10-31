"""Train the transition-aware retrieval model from cached V-JEPA features."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    ANNOTATION_ROOT,
    CHECKPOINT_ROOT,
    FEATURE_ROOT,
    RANDOM_SEED,
    TEXT_MODEL_NAME,
)
from src.models.transition_model.encoder import encode_text_queries
from src.models.transition_model.pipelines.evaluate import encode_row_texts, evaluate_rows
from src.models.transition_model.retrieval import TransitionRetrievalModel, save_checkpoint
from src.models.transition_model.utils import (
    TransitionDataset,
    collate_transitions,
    ensure_dir,
    get_device,
    load_split_rows,
    load_transitions,
    set_random_seed,
)


# Config

BATCH_SIZE = 128
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
NUM_WORKERS = 0
SELECTION_METRIC = "mrr"
TEXT_ENCODER = "minilm"
CHECKPOINT_NAME = "transition_model_best.pt"
LOG_NAME = "train_log.jsonl"


# Sub functions

def contrastive_loss(scores: torch.Tensor, text_ids: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE where every pair sharing the same query text is positive.

    Something-Something queries are templated, so a batch often holds several
    videos with identical text; treating them as negatives would be wrong.
    """
    positives = text_ids[:, None] == text_ids[None, :]
    masked = scores.masked_fill(~positives, float("-inf"))
    text_to_video = scores.logsumexp(dim=1) - masked.logsumexp(dim=1)
    video_to_text = scores.logsumexp(dim=0) - masked.logsumexp(dim=0)
    return (text_to_video.mean() + video_to_text.mean()) / 2


def train_one_epoch(
    model: TransitionRetrievalModel,
    loader: DataLoader,
    text_bank: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Run one training epoch and return the mean loss."""
    model.train()
    device = text_bank.device
    total, batches = 0.0, 0
    for transitions, mask, text_ids in tqdm(loader, desc="Training", leave=False):
        transitions, mask, text_ids = transitions.to(device), mask.to(device), text_ids.to(device)
        videos = model.encode_videos(transitions, mask)
        texts = model.encode_texts(text_bank[text_ids])
        loss = contrastive_loss(model.score(texts, videos), text_ids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
        batches += 1
    return total / max(batches, 1)


def build_train_loader(
    rows: list[dict],
    feature_dir: Path,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, list[str]]:
    """Return a shuffled loader of (transitions, text id) and the unique texts."""
    texts = list(dict.fromkeys(row["text_query"] for row in rows))
    text_index = {text: i for i, text in enumerate(texts)}
    dataset = TransitionDataset(
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


# Main function

def train_transition_model(
    train_annotations: Path,
    val_annotations: Path,
    feature_dir: Path,
    checkpoint_dir: Path,
    *,
    epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    query_conditioned: bool = True,
    text_encoder: str = TEXT_ENCODER,
    text_model_name: str = TEXT_MODEL_NAME,
    device: str | None = None,
    limit: int | None = None,
    seed: int = RANDOM_SEED,
) -> Path:
    """Train the transition-aware model and return the best checkpoint path."""
    set_random_seed(seed)
    device = get_device(device)
    checkpoint_dir = ensure_dir(checkpoint_dir)
    best_path = checkpoint_dir / CHECKPOINT_NAME
    log_path = checkpoint_dir / LOG_NAME

    train_rows, train_skipped = load_split_rows(train_annotations, feature_dir, limit)
    val_rows, val_skipped = load_split_rows(val_annotations, feature_dir, limit)
    if not train_rows or not val_rows:
        raise ValueError(f"train and val need rows with cached features in {feature_dir}")
    print(f"Train rows: {len(train_rows)} (skipped {train_skipped} without features)")
    print(f"Val rows: {len(val_rows)} (skipped {val_skipped} without features)")

    loader, train_texts = build_train_loader(train_rows, feature_dir, batch_size, seed)
    text_bank = torch.from_numpy(
        encode_text_queries(
            train_texts, encoder_name=text_encoder, model_name=text_model_name, device=device
        )
    ).to(device)
    val_texts = encode_row_texts(val_rows, text_encoder, text_model_name, device)

    video_dim = load_transitions(feature_dir, str(train_rows[0]["target_video_id"])).shape[1]
    model = TransitionRetrievalModel(
        video_dim, text_bank.shape[1], query_conditioned=query_conditioned
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY
    )

    best_score = float("-inf")
    log_path.write_text("", encoding="utf-8")
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, loader, text_bank, optimizer)
        metrics = evaluate_rows(model, val_rows, val_texts, feature_dir)
        record = {"epoch": epoch, "train_loss": loss, "val": metrics}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(
            f"Epoch {epoch}: loss={loss:.4f} "
            f"val R@1={metrics['recall@1']:.4f} MRR={metrics['mrr']:.4f}"
        )

        if metrics[SELECTION_METRIC] > best_score:
            best_score = metrics[SELECTION_METRIC]
            save_checkpoint(
                model,
                best_path,
                epoch=epoch,
                val_metrics=metrics,
                selection_metric=SELECTION_METRIC,
                text_encoder=text_encoder,
                text_model_name=text_model_name,
                feature_dir=str(feature_dir),
                seed=seed,
            )
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-annotations", type=Path, default=ANNOTATION_ROOT / "train.jsonl")
    parser.add_argument("--val-annotations", type=Path, default=ANNOTATION_ROOT / "val.jsonl")
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_ROOT / "vjepa")
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--no-query-conditioning", action="store_true")
    parser.add_argument("--text-encoder", choices=("minilm", "mock"), default=TEXT_ENCODER)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    path = train_transition_model(
        args.train_annotations,
        args.val_annotations,
        args.feature_dir,
        args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        query_conditioned=not args.no_query_conditioning,
        text_encoder=args.text_encoder,
        device=args.device,
        limit=args.limit,
        seed=args.seed,
    )
    print(f"Best checkpoint: {path}")


if __name__ == "__main__":
    main()
