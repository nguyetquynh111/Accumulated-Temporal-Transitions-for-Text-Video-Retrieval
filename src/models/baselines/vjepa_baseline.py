"""B2 baseline (pooled V-JEPA, no transitions) and the transition ablations (T2.7, T3.3).

Every variant feeds a different view of the same cached V-JEPA states into the
shared TransitionRetrievalModel, trains it with the main training protocol, and
is scored with the shared evaluator. Only the video input, and for
two variants the query-conditioned reranker, changes between runs:

- pooled: mean state as one token, no reranker (B2).
- full_transition: [z[t], z[t + 1] - z[t]] tokens with the reranker. This is the
  main model; evaluate its checkpoint with it to get comparable rankings.
- no_transition: the raw state sequence z[1:T] instead of transitions. This is
  the matched control for the main model.
- transition_only: the transitions z[t + 1] - z[t] alone, without states.
- shuffled_states: no_transition with the state order permuted per video. The
  permutation is redrawn every time a training video is loaded (so every
  epoch) and is fixed per video id in evaluation.
- mean_transition: the mean transition vector as one token.
- final_minus_initial: z[T] - z[1] as one token.
- full_transition_without_query_conditioning: [z, dz] tokens, no reranker (B3).

Note that the mean of z[t + 1] - z[t] telescopes to (z[T] - z[1]) / (T - 1), and
the model's input LayerNorm removes that scale, so mean_transition and
final_minus_initial see the same input up to a constant factor.

With --reverse-time, trained checkpoints are re-evaluated with each video's
states reversed in time before the variant input (and its transitions) is built.
"""

import argparse
import json
from pathlib import Path
import zlib

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import (
    ANNOTATION_ROOT,
    CHECKPOINT_ROOT,
    FEATURE_ROOT,
    LOG_ROOT,
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
from src.models.transition_model.pipelines.train import (
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    PATIENCE,
    RUN_NAME as MAIN_RUN_NAME,
    SELECTION_METRIC,
    SELECTION_METRICS,
    WEIGHT_DECAY,
    checkpoint_path,
    fit_retrieval_model,
    log_path,
)
from src.models.transition_model.pipelines.train_token_reranker import per_query_records
from src.models.transition_model.retrieval import (
    TransitionRetrievalModel,
    load_checkpoint,
    score_transition_retrieval,
)
from src.models.transition_model.transition import compute_transitions
from src.models.transition_model.utils import (
    collate_transitions,
    encode_row_texts,
    ensure_dir,
    feature_path,
    get_device,
    load_split_rows,
    load_states,
    pad_sequences,
    set_random_seed,
    unique_in_order,
)


# Config

# Variant name -> whether the query-conditioned reranker is used.
VARIANTS = {
    "pooled": False,
    "full_transition": True,
    "no_transition": True,
    "transition_only": True,
    "shuffled_states": True,
    "mean_transition": True,
    "final_minus_initial": True,
    "full_transition_without_query_conditioning": False,
}
STATE_TRANSITION_VARIANTS = ("full_transition", "full_transition_without_query_conditioning")
ABLATION_VARIANTS = (
    "no_transition",
    "transition_only",
    "shuffled_states",
    "mean_transition",
    "final_minus_initial",
    "full_transition_without_query_conditioning",
)
ENCODE_BATCH_SIZE = 256
NUM_WORKERS = 0
TEXT_ENCODER = "minilm"
ACTION_CONTRAST_FILE = "action_contrast.jsonl"
# Base seed of the fixed per-video evaluation permutation of shuffled_states.
SHUFFLE_EVAL_SEED = 0
REVERSED_SUFFIX = "_reversed"


# Sub functions

def result_name(variant: str) -> str:
    """Return the model name used in result, ranking, and checkpoint files."""
    if variant == "pooled":
        return "vjepa_baseline"
    if variant == "full_transition":
        return MAIN_RUN_NAME
    return f"ablation_{variant}"


def variant_video_input(variant: str) -> str:
    """Return the TransitionRetrievalModel video_input a variant is trained with."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    return "state_transition" if variant in STATE_TRANSITION_VARIANTS else "transition"


def fixed_permutation(video_id: str, length: int, seed: int = SHUFFLE_EVAL_SEED) -> np.ndarray:
    """Return the evaluation permutation of one video's states; stable across runs."""
    rng = np.random.default_rng([seed, zlib.crc32(str(video_id).encode("utf-8"))])
    return rng.permutation(length)


def prepare_states(
    variant: str,
    states: np.ndarray,
    video_id: str,
    *,
    reverse_time: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply time reversal and, for shuffled_states, the state permutation.

    With rng a fresh permutation is drawn (training); without it the fixed
    per-video permutation is used (evaluation).
    """
    states = np.asarray(states, dtype=np.float32)
    if reverse_time:
        states = states[::-1]
    if variant == "shuffled_states":
        if rng is not None:
            order = rng.permutation(len(states))
        else:
            order = fixed_permutation(video_id, len(states))
        states = states[order]
    return np.ascontiguousarray(states)


def main_checkpoint_path(checkpoint_dir: Path = CHECKPOINT_ROOT, seed: int = RANDOM_SEED) -> Path:
    """Return the main model's best checkpoint for one seed, as pipelines.train saves it."""
    return checkpoint_path(checkpoint_dir, MAIN_RUN_NAME, seed)


def variant_sequence(
    variant: str,
    state_embeddings: np.ndarray,
    transition_embeddings: np.ndarray | None = None,
) -> np.ndarray:
    """Return the (L, D) or (L, 2D) video input one variant feeds to the retrieval model."""
    states = np.asarray(state_embeddings, dtype=np.float32)
    if states.ndim != 2 or states.shape[0] < 2:
        raise ValueError("state_embeddings must have shape (T, D) with T >= 2")
    if variant == "pooled":
        return states.mean(axis=0, keepdims=True)
    if variant in ("no_transition", "shuffled_states"):
        return states
    if variant == "final_minus_initial":
        return (states[-1] - states[0])[None]
    transitions = (
        compute_transitions(states)
        if transition_embeddings is None
        else np.asarray(transition_embeddings, dtype=np.float32)
    )
    if variant in STATE_TRANSITION_VARIANTS:
        return np.concatenate((states[:-1], transitions), axis=1).astype(np.float32)
    if variant == "transition_only":
        return transitions
    if variant == "mean_transition":
        return transitions.mean(axis=0, keepdims=True)
    raise ValueError(f"unknown variant: {variant}")


class VariantDataset(Dataset):
    """Yield (variant sequence, label) pairs, reading feature caches lazily.

    shuffle_seed puts shuffled_states in training mode: a new permutation is
    drawn on every load. Without it the fixed evaluation permutation is used.
    """

    def __init__(
        self,
        variant: str,
        video_ids: list[str],
        labels: list[int],
        feature_dir: Path,
        *,
        reverse_time: bool = False,
        shuffle_seed: int | None = None,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant: {variant}")
        if len(video_ids) != len(labels):
            raise ValueError("video_ids and labels must have the same length")
        self.variant = variant
        self.video_ids = video_ids
        self.labels = labels
        self.feature_dir = Path(feature_dir)
        self.reverse_time = reverse_time
        self.rng = None if shuffle_seed is None else np.random.default_rng(shuffle_seed)

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        video_id = self.video_ids[index]
        states = prepare_states(
            self.variant,
            load_states(self.feature_dir, video_id),
            video_id,
            reverse_time=self.reverse_time,
            rng=self.rng,
        )
        return variant_sequence(self.variant, states), self.labels[index]


def encode_variant_bank(
    model: TransitionRetrievalModel,
    variant: str,
    video_ids: list[str],
    feature_dir: Path,
    batch_size: int = ENCODE_BATCH_SIZE,
    reverse_time: bool = False,
) -> np.ndarray:
    """Return encoded video representations (N, model_dim) in video_ids order."""
    dataset = VariantDataset(
        variant, video_ids, list(range(len(video_ids))), feature_dir, reverse_time=reverse_time
    )
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
    reverse_time: bool = False,
) -> tuple[np.ndarray, list[str], dict]:
    """Score every query against the split's target videos; return scores, candidates, metrics."""
    candidates = candidate_ids(rows)
    videos = encode_variant_bank(
        model, variant, candidates, feature_dir, reverse_time=reverse_time
    )
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
    texts = unique_in_order(row["text_query"] for row in rows)
    text_index = {text: i for i, text in enumerate(texts)}
    dataset = VariantDataset(
        variant,
        [str(row["target_video_id"]) for row in rows],
        [text_index[row["text_query"]] for row in rows],
        feature_dir,
        shuffle_seed=seed if variant == "shuffled_states" else None,
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


def evaluate_candidate_file(
    model: TransitionRetrievalModel,
    variant: str,
    path: Path | None,
    rows: list[dict],
    texts: np.ndarray,
    feature_dir: Path,
    reverse_time: bool = False,
    ranks_out: dict[str, int] | None = None,
) -> tuple[dict | None, int]:
    """Score one hard-negative-schema file; (None, 0) when it is missing or empty.

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
    videos = encode_variant_bank(model, variant, video_ids, feature_dir, reverse_time=reverse_time)
    metrics = evaluate_hard_negatives(
        kept,
        rows,
        texts,
        video_ids,
        videos,
        lambda text, candidates: score_transition_retrieval(text, candidates, model),
        ranks_out=ranks_out,
    )
    return metrics, skipped


# Main functions

def train_variant_model(
    variant: str,
    train_annotations: Path,
    val_annotations: Path,
    feature_dir: Path,
    checkpoint_dir: Path,
    *,
    log_dir: Path = LOG_ROOT,
    epochs: int = NUM_EPOCHS,
    patience: int | None = PATIENCE,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    selection_metric: str = SELECTION_METRIC,
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
    video_input = variant_video_input(variant)
    if selection_metric not in SELECTION_METRICS:
        raise ValueError(f"selection_metric must be one of {SELECTION_METRICS}")
    set_random_seed(seed)
    device = get_device(device)
    checkpoint_dir = ensure_dir(checkpoint_dir)
    name = result_name(variant)

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
        video_dim,
        text_bank.shape[1],
        query_conditioned=VARIANTS[variant],
        video_input=video_input,
    ).to(device)

    best_path = checkpoint_path(checkpoint_dir, name, seed)
    fit_retrieval_model(
        model,
        loader,
        text_bank,
        lambda m: score_variant_rows(m, variant, val_rows, val_texts, feature_dir)[2],
        best_path,
        log_path(log_dir, name, seed),
        epochs=epochs,
        patience=patience,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        selection_metric=selection_metric,
        checkpoint_metadata={
            "variant": variant,
            "text_encoder": text_encoder,
            "text_model_name": text_model_name,
            "feature_dir": str(feature_dir),
            "seed": seed,
            "train_config": {
                "run_name": name,
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
            },
        },
        log_prefix=f"[{name}] ",
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
    reverse_time: bool = False,
) -> np.ndarray:
    """Return score matrix for one ablation variant.

    Candidates are the keys of state_embeddings, in order. Without a model,
    each video input is mean-pooled and dot-scored, which needs matching text
    and video dimensions; with the variant's trained model, text_embeddings are
    raw text-encoder outputs. Cached transitions are ignored when the states
    are reordered (reverse_time or shuffled_states), since they no longer match.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    reorders = reverse_time or variant == "shuffled_states"
    sequences = [
        variant_sequence(
            variant,
            prepare_states(variant, states, video_id, reverse_time=reverse_time),
            None if reorders else transition_embeddings.get(video_id),
        )
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
    action_contrast: Path | None = None,
    rankings_path: Path | None = None,
    device: str | None = None,
    limit: int | None = None,
    reverse_time: bool = False,
    per_query_path: Path | None = None,
) -> dict:
    """Evaluate one variant checkpoint on full, hard-negative, and action-contrast retrieval.

    With reverse_time, every video's states are reversed before its input is built.
    per_query_path, if given, gets one line per query with the target rank on the
    full split, the hard-negative set, and the action-contrast set (null when the
    query is not in that set), the format T3.2's paired bootstrap reads.
    """
    device = get_device(device)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    trained_variant = checkpoint.get("variant")
    if trained_variant is not None and trained_variant != variant:
        raise ValueError(
            f"checkpoint was trained as variant {trained_variant!r}, which does not match {variant}"
        )
    if checkpoint["model_config"]["query_conditioned"] != VARIANTS[variant]:
        raise ValueError(f"checkpoint reranker setting does not match variant {variant}")
    if model.video_input != variant_video_input(variant):
        raise ValueError(
            f"checkpoint video_input {model.video_input!r} does not match variant {variant} "
            f"({variant_video_input(variant)!r})"
        )
    rows, skipped = load_split_rows(annotations, feature_dir, limit)
    if not rows:
        raise ValueError(f"no annotation rows with cached features in {feature_dir}")
    texts = encode_row_texts(rows, checkpoint["text_encoder"], checkpoint["text_model_name"], device)
    scores, candidates, full = score_variant_rows(
        model, variant, rows, texts, feature_dir, reverse_time
    )

    name = result_name(variant)
    summary = checkpoint.get("training_summary") or {}
    result = build_result(
        name,
        rows,
        full,
        variant=variant,
        checkpoint_path=str(checkpoint_path),
        seed=checkpoint.get("seed"),
        best_epoch=summary.get("best_epoch", checkpoint.get("epoch")),
        epochs_run=summary.get("epochs_run"),
        training_summary=summary or None,
        video_input=model.video_input,
        query_conditioned=VARIANTS[variant],
        reverse_time=reverse_time,
        skipped_missing_features=skipped,
    )
    set_ranks = {}
    for key, path in (("hard_negative", hard_negatives), ("action_contrast", action_contrast)):
        set_ranks[key] = {}
        metrics, set_skipped = evaluate_candidate_file(
            model, variant, path, rows, texts, feature_dir, reverse_time, set_ranks[key]
        )
        result[key] = metrics
        result[f"{key}_skipped"] = set_skipped

    save_metrics(result, output_path)
    if rankings_path is not None:
        save_jsonl(ranking_records(name, rows, scores, candidates), rankings_path)
    if per_query_path is not None:
        save_jsonl(per_query_records(rows, scores, candidates, set_ranks), per_query_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        nargs="+",
        choices=(*VARIANTS, "all_ablations"),
        default=["pooled"],
        help="one or more variants; all_ablations runs the six T3.3 variants",
    )
    parser.add_argument("--train-annotations", type=Path, default=ANNOTATION_ROOT / "train.jsonl")
    parser.add_argument("--val-annotations", type=Path, default=ANNOTATION_ROOT / "val.jsonl")
    parser.add_argument("--test-annotations", type=Path, default=ANNOTATION_ROOT / "test.jsonl")
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_ROOT / "vjepa")
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--log-dir", type=Path, default=LOG_ROOT)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="evaluate this checkpoint instead of training (one variant and one seed only)",
    )
    parser.add_argument(
        "--hard-negatives", type=Path, default=ANNOTATION_ROOT / "hard_negatives.jsonl"
    )
    parser.add_argument(
        "--action-contrast", type=Path, default=ANNOTATION_ROOT / ACTION_CONTRAST_FILE
    )
    parser.add_argument("--result-dir", type=Path, default=RESULT_ROOT)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="epoch limit")
    parser.add_argument(
        "--patience",
        type=int,
        default=PATIENCE,
        help="stop after this many epochs without validation gain; 0 disables",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument(
        "--selection-metric", choices=SELECTION_METRICS, default=SELECTION_METRIC
    )
    parser.add_argument("--text-encoder", choices=("minilm", "mock"), default=TEXT_ENCODER)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[RANDOM_SEED],
        help="one training run (or main checkpoint) per seed",
    )
    parser.add_argument(
        "--reverse-time",
        action="store_true",
        help="re-evaluate trained checkpoints with states reversed in time (no training)",
    )
    args = parser.parse_args()

    variants = []
    for variant in args.variant:
        variants.extend(ABLATION_VARIANTS if variant == "all_ablations" else [variant])
    variants = list(dict.fromkeys(variants))
    if args.checkpoint is not None and (len(variants) != 1 or len(args.seeds) != 1):
        parser.error("--checkpoint needs exactly one variant and one seed")

    for variant in variants:
        for seed in args.seeds:
            checkpoint = args.checkpoint
            if checkpoint is None and variant == "full_transition":
                checkpoint = main_checkpoint_path(args.checkpoint_dir, seed)
                if not checkpoint.exists():
                    parser.error(f"missing main checkpoint: {checkpoint}")
            if checkpoint is None and args.reverse_time:
                checkpoint = checkpoint_path(args.checkpoint_dir, result_name(variant), seed)
                if not checkpoint.exists():
                    parser.error(f"--reverse-time needs a trained checkpoint: {checkpoint}")
            if checkpoint is None:
                checkpoint = train_variant_model(
                    variant,
                    args.train_annotations,
                    args.val_annotations,
                    args.feature_dir,
                    args.checkpoint_dir,
                    log_dir=args.log_dir,
                    epochs=args.epochs,
                    patience=args.patience or None,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    selection_metric=args.selection_metric,
                    text_encoder=args.text_encoder,
                    device=args.device,
                    limit=args.limit,
                    seed=seed,
                )
            suffix = REVERSED_SUFFIX if args.reverse_time else ""
            stem = f"{result_name(variant)}_seed{seed}{suffix}"
            output_stem = (
                f"full_transition_seed{seed}{suffix}" if variant == "full_transition" else stem
            )
            output = args.result_dir / f"{output_stem}.json"
            result = evaluate_variant(
                variant,
                checkpoint,
                args.test_annotations,
                args.feature_dir,
                output,
                hard_negatives=args.hard_negatives,
                action_contrast=args.action_contrast,
                rankings_path=args.result_dir / f"rankings_{stem}.jsonl",
                device=args.device,
                limit=args.limit,
                reverse_time=args.reverse_time,
                per_query_path=args.result_dir / f"per_query_{stem}.jsonl",
            )
            summary = {
                key: result[key]
                and {k: v for k, v in result[key].items() if k not in ("by_negative_type", "coverage")}
                for key in ("full", "hard_negative", "action_contrast")
            }
            print(json.dumps({stem: summary}, indent=2))
            print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
