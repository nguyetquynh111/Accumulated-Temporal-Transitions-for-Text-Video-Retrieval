"""Train the transition-aware retrieval model from cached V-JEPA features (T2.4).

The model reads [z, dz] tokens by default (--video-input state_transition).
Training follows the shared protocol in config.py: early stopping on
validation MRR with a fixed epoch limit, and one run per seed. Checkpoints
(--checkpoint-dir) and per-epoch logs (--log-dir) are named <run_name>_seed<seed>_*.
"""

import argparse
import json
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    ANNOTATION_ROOT,
    CHECKPOINT_ROOT,
    EARLY_STOPPING_PATIENCE,
    LOG_ROOT,
    RANDOM_SEED,
    TEXT_MODEL_NAME,
    TRAIN_BATCH_SIZE,
    TRAIN_LEARNING_RATE,
    TRAIN_MAX_EPOCHS,
    TRAIN_SELECTION_METRIC,
    TRAIN_WEIGHT_DECAY,
    VJEPA_FEATURE_ROOT,
)
from src.models.transition_model.encoder import TEXT_ENCODER_NAMES, encode_text_queries
from src.models.transition_model.pipelines.evaluate import evaluate_rows
from src.models.transition_model.retrieval import (
    RERANK_HIDDEN_DIM,
    TransitionRetrievalModel,
    save_checkpoint,
)
from src.models.transition_model.transition import (
    DEFAULT_VIDEO_INPUT,
    DROPOUT,
    MODEL_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    VIDEO_INPUTS,
)
from src.models.transition_model.utils import (
    TransitionDataset,
    collate_transitions,
    encode_row_texts,
    ensure_dir,
    get_device,
    load_split_rows,
    load_states,
    set_random_seed,
    unique_in_order,
)


# Config

RUN_NAME = "transition_model"
BATCH_SIZE = TRAIN_BATCH_SIZE
NUM_EPOCHS = TRAIN_MAX_EPOCHS
PATIENCE = EARLY_STOPPING_PATIENCE
LEARNING_RATE = TRAIN_LEARNING_RATE
WEIGHT_DECAY = TRAIN_WEIGHT_DECAY
NUM_WORKERS = 0
SELECTION_METRICS = ("mrr", "recall@1")
SELECTION_METRIC = TRAIN_SELECTION_METRIC
TEXT_ENCODER = "minilm"


# Sub functions

def run_stem(run_name: str = RUN_NAME, seed: int = RANDOM_SEED) -> str:
    """Return the file-name prefix of one training run, e.g. transition_model_seed42."""
    return f"{run_name}_seed{seed}"


def checkpoint_path(
    checkpoint_dir: Path, run_name: str = RUN_NAME, seed: int = RANDOM_SEED
) -> Path:
    """Return where the best checkpoint of one training run is saved."""
    return Path(checkpoint_dir) / f"{run_stem(run_name, seed)}_best.pt"


def log_path(log_dir: Path, run_name: str = RUN_NAME, seed: int = RANDOM_SEED) -> Path:
    """Return where the per-epoch training log of one run is saved."""
    return Path(log_dir) / f"{run_stem(run_name, seed)}_train_log.jsonl"


class EarlyStopping:
    """Track the best validation score and stop after `patience` epochs without gain.

    patience=None disables stopping, so training runs to the epoch limit.
    """

    def __init__(self, patience: int | None = PATIENCE):
        if patience is not None and patience < 1:
            raise ValueError("patience must be positive or None")
        self.patience = patience
        self.best_score = float("-inf")
        self.best_epoch = 0
        self.last_epoch = 0

    def update(self, epoch: int, score: float) -> bool:
        """Record one epoch's score and return True when it is a new best."""
        self.last_epoch = epoch
        if score > self.best_score:
            self.best_score, self.best_epoch = score, epoch
            return True
        return False

    @property
    def should_stop(self) -> bool:
        return (
            self.patience is not None
            and self.last_epoch - self.best_epoch >= self.patience
        )

    def summary(self, max_epochs: int) -> dict:
        """Return how the run ended; a run cut off while improving is flagged."""
        stopped_early = self.should_stop
        return {
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "epochs_run": self.last_epoch,
            "max_epochs": max_epochs,
            "patience": self.patience,
            "stopped_early": stopped_early,
            # Reached the epoch limit before `patience` epochs passed without gain.
            "reached_epoch_limit_while_improving": not stopped_early
            and self.last_epoch == max_epochs,
        }


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
    for tokens, mask, text_ids in tqdm(loader, desc="Training", leave=False):
        tokens, mask, text_ids = tokens.to(device), mask.to(device), text_ids.to(device)
        videos = model.encode_videos(tokens, mask)
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
    num_workers: int = NUM_WORKERS,
    video_input: str = DEFAULT_VIDEO_INPUT,
) -> tuple[DataLoader, list[str]]:
    """Return a shuffled loader of (video tokens, text id) and the unique texts."""
    texts = unique_in_order(row["text_query"] for row in rows)
    text_index = {text: i for i, text in enumerate(texts)}
    dataset = TransitionDataset(
        [str(row["target_video_id"]) for row in rows],
        [text_index[row["text_query"]] for row in rows],
        feature_dir,
        video_input,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(dataset) > batch_size,
        num_workers=num_workers,
        collate_fn=collate_transitions,
        generator=torch.Generator().manual_seed(seed),
    )
    return loader, texts


def fit_retrieval_model(
    model: TransitionRetrievalModel,
    loader: DataLoader,
    text_bank: torch.Tensor,
    validate: Callable[[TransitionRetrievalModel], dict],
    best_path: Path,
    train_log: Path,
    *,
    epochs: int = NUM_EPOCHS,
    patience: int | None = PATIENCE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    selection_metric: str = SELECTION_METRIC,
    checkpoint_metadata: dict | None = None,
    log_prefix: str = "",
) -> dict:
    """Run the shared training protocol and return its summary (T2.4).

    Every trained model, the main one and each ablation, goes through this
    loop: AdamW, validation with `validate` after every epoch, the best
    `selection_metric` checkpoint kept at `best_path`, and early stopping after
    `patience` epochs without gain or at `epochs`. The summary is also stored
    in the checkpoint as `training_summary`.
    """
    if selection_metric not in SELECTION_METRICS:
        raise ValueError(f"selection_metric must be one of {SELECTION_METRICS}")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    stopper = EarlyStopping(patience)
    train_log = Path(train_log)
    train_log.parent.mkdir(parents=True, exist_ok=True)
    train_log.write_text("", encoding="utf-8")
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, loader, text_bank, optimizer)
        metrics = validate(model)
        with train_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"epoch": epoch, "train_loss": loss, "val": metrics}) + "\n")
        print(
            f"{log_prefix}Epoch {epoch}: loss={loss:.4f} "
            f"val R@1={metrics['recall@1']:.4f} MRR={metrics['mrr']:.4f}"
        )
        if stopper.update(epoch, metrics[selection_metric]):
            save_checkpoint(
                model,
                best_path,
                epoch=epoch,
                val_metrics=metrics,
                selection_metric=selection_metric,
                **(checkpoint_metadata or {}),
            )
        if stopper.should_stop:
            print(f"{log_prefix}Early stop at epoch {epoch}: no gain for {patience} epochs")
            break

    summary = stopper.summary(epochs)
    checkpoint = torch.load(best_path, map_location="cpu")
    checkpoint["training_summary"] = summary
    torch.save(checkpoint, best_path)
    if summary["reached_epoch_limit_while_improving"]:
        print(
            f"{log_prefix}Warning: {Path(best_path).name} hit the {epochs}-epoch limit "
            "while still improving"
        )
    print(f"{log_prefix}Best epoch {summary['best_epoch']} of {summary['epochs_run']}")
    return summary


# Main function

def train_transition_model(
    train_annotations: Path,
    val_annotations: Path,
    feature_dir: Path,
    checkpoint_dir: Path,
    *,
    log_dir: Path = LOG_ROOT,
    run_name: str = RUN_NAME,
    epochs: int = NUM_EPOCHS,
    patience: int | None = PATIENCE,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    selection_metric: str = SELECTION_METRIC,
    query_conditioned: bool = True,
    model_dim: int = MODEL_DIM,
    num_heads: int = NUM_HEADS,
    num_layers: int = NUM_LAYERS,
    dropout: float = DROPOUT,
    rerank_hidden_dim: int = RERANK_HIDDEN_DIM,
    video_input: str = DEFAULT_VIDEO_INPUT,
    text_encoder: str = TEXT_ENCODER,
    text_model_name: str = TEXT_MODEL_NAME,
    num_workers: int = NUM_WORKERS,
    device: str | None = None,
    limit: int | None = None,
    seed: int = RANDOM_SEED,
) -> Path:
    """Train the transition-aware model and return the best checkpoint path.

    The checkpoint with the best validation `selection_metric` is kept at
    `<checkpoint_dir>/<run_name>_seed<seed>_best.pt`; the per-epoch JSONL log is
    `<log_dir>/<run_name>_seed<seed>_train_log.jsonl`. Training stops after `patience` epochs without improvement or at
    `epochs`; the checkpoint's `training_summary` records how the run ended.
    """
    if selection_metric not in SELECTION_METRICS:
        raise ValueError(f"selection_metric must be one of {SELECTION_METRICS}")
    if video_input not in VIDEO_INPUTS:
        raise ValueError(f"video_input must be one of {VIDEO_INPUTS}")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    train_config = {
        "run_name": run_name,
        "epochs": epochs,
        "patience": patience,
        "video_input": video_input,
        "seed": seed,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "selection_metric": selection_metric,
        "train_annotations": str(train_annotations),
        "val_annotations": str(val_annotations),
        "limit": limit,
    }
    set_random_seed(seed)
    device = get_device(device)
    checkpoint_dir = ensure_dir(checkpoint_dir)
    best_path = checkpoint_path(checkpoint_dir, run_name, seed)
    train_log = log_path(log_dir, run_name, seed)

    train_rows, train_skipped = load_split_rows(train_annotations, feature_dir, limit)
    val_rows, val_skipped = load_split_rows(val_annotations, feature_dir, limit)
    if not train_rows or not val_rows:
        raise ValueError(f"train and val need rows with cached features in {feature_dir}")
    print(f"Train rows: {len(train_rows)} (skipped {train_skipped} without features)")
    print(f"Val rows: {len(val_rows)} (skipped {val_skipped} without features)")

    loader, train_texts = build_train_loader(
        train_rows, feature_dir, batch_size, seed, num_workers, video_input
    )
    text_bank = torch.from_numpy(
        encode_text_queries(
            train_texts, encoder_name=text_encoder, model_name=text_model_name, device=device
        )
    ).to(device)
    val_texts = encode_row_texts(val_rows, text_encoder, text_model_name, device)

    video_dim = load_states(feature_dir, str(train_rows[0]["target_video_id"])).shape[1]
    model = TransitionRetrievalModel(
        video_dim,
        text_bank.shape[1],
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        rerank_hidden_dim=rerank_hidden_dim,
        query_conditioned=query_conditioned,
        video_input=video_input,
    ).to(device)
    fit_retrieval_model(
        model,
        loader,
        text_bank,
        lambda m: evaluate_rows(m, val_rows, val_texts, feature_dir, num_workers=num_workers),
        best_path,
        train_log,
        epochs=epochs,
        patience=patience,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        selection_metric=selection_metric,
        checkpoint_metadata={
            "text_encoder": text_encoder,
            "text_model_name": text_model_name,
            "feature_dir": str(feature_dir),
            "seed": seed,
            "train_config": train_config,
        },
    )
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--train-annotations", type=Path, default=ANNOTATION_ROOT / "train.jsonl")
    parser.add_argument("--val-annotations", type=Path, default=ANNOTATION_ROOT / "val.jsonl")
    parser.add_argument("--feature-dir", type=Path, default=VJEPA_FEATURE_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--log-dir", type=Path, default=LOG_ROOT)
    parser.add_argument(
        "--run-name",
        default=RUN_NAME,
        help="prefix of the checkpoint and log names; the seed is appended",
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="epoch limit")
    parser.add_argument(
        "--patience",
        type=int,
        default=PATIENCE,
        help="stop after this many epochs without validation gain; 0 disables",
    )
    parser.add_argument(
        "--video-input",
        choices=VIDEO_INPUTS,
        default=DEFAULT_VIDEO_INPUT,
        help="state_transition: [z, dz] tokens (main model); transition: dz only (ablation)",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument(
        "--selection-metric", choices=SELECTION_METRICS, default=SELECTION_METRIC
    )
    parser.add_argument(
        "--no-query-conditioning",
        action="store_true",
        help="drop the query-conditioned reranker (give this run its own --run-name)",
    )
    parser.add_argument("--model-dim", type=int, default=MODEL_DIM)
    parser.add_argument("--num-heads", type=int, default=NUM_HEADS)
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--rerank-hidden-dim", type=int, default=RERANK_HIDDEN_DIM)
    parser.add_argument("--text-encoder", choices=TEXT_ENCODER_NAMES, default=TEXT_ENCODER)
    parser.add_argument("--text-model-name", default=TEXT_MODEL_NAME)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    path = train_transition_model(
        args.train_annotations,
        args.val_annotations,
        args.feature_dir,
        args.checkpoint_dir,
        log_dir=args.log_dir,
        run_name=args.run_name,
        epochs=args.epochs,
        patience=args.patience or None,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        selection_metric=args.selection_metric,
        query_conditioned=not args.no_query_conditioning,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        rerank_hidden_dim=args.rerank_hidden_dim,
        video_input=args.video_input,
        text_encoder=args.text_encoder,
        text_model_name=args.text_model_name,
        num_workers=args.num_workers,
        device=args.device,
        limit=args.limit,
        seed=args.seed,
    )
    print(f"Best checkpoint: {path}")


if __name__ == "__main__":
    main()
