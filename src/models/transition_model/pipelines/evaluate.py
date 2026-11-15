"""Evaluate a trained transition model on full and hard-negative retrieval (T3.1).

Each checkpoint is scored on:
- full retrieval over the split's target videos;
- the hard-negative set, in aggregate and per negative type (pairwise
  accuracy, and Recall@1 against the target and only that type's negatives),
  with its coverage;
- the action-contrast set, whose negatives are all opposite actions, when its
  file exists (built by T2.6).

Without --checkpoint, every seed in --seeds is evaluated from
<run_name>_seed<seed>_best.pt and saved to <run_name>_seed<seed>.json.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import (
    ANNOTATION_ROOT,
    CHECKPOINT_ROOT,
    RESULT_ROOT,
    TRAIN_SEEDS,
    VJEPA_FEATURE_ROOT,
)
from src.eval.evaluate import (
    compute_retrieval_metrics,
    save_metrics,
    summarize_candidate_set,
)
from src.models.transition_model.retrieval import (
    ENCODE_BATCH_SIZE,
    TransitionRetrievalModel,
    encode_video_bank,
    load_checkpoint,
    score_transition_retrieval,
)
from src.models.transition_model.transition import save_transition_feature_cache
from src.models.transition_model.utils import (
    candidate_video_ids,
    encode_row_texts,
    feature_path,
    get_device,
    load_jsonl,
    load_split_rows,
    load_states,
    unique_in_order,
)


# Config

MODEL_NAME = "transition_model"
BATCH_SIZE = ENCODE_BATCH_SIZE
NUM_WORKERS = 0
HARD_NEGATIVE_FILE = "hard_negatives.jsonl"
ACTION_CONTRAST_FILE = "action_contrast.jsonl"


# Sub functions

def evaluate_rows(
    model: TransitionRetrievalModel,
    rows: list[dict],
    text_embeddings: np.ndarray,
    feature_dir: Path,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> dict:
    """Score every query against the split's target videos with the shared evaluator."""
    candidate_ids = candidate_video_ids(rows)
    videos = encode_video_bank(model, candidate_ids, feature_dir, batch_size, num_workers)
    scores = score_transition_retrieval(text_embeddings, videos, model)
    return compute_retrieval_metrics(
        scores,
        [str(row["query_id"]) for row in rows],
        candidate_ids,
        [str(row["target_video_id"]) for row in rows],
    )


def score_candidate_sets(
    model: TransitionRetrievalModel,
    rows: list[dict],
    text_embeddings: np.ndarray,
    candidate_rows: list[dict],
    feature_dir: Path,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> tuple[list[dict], dict, int]:
    """Score each candidate-set row (hard-negative schema) against its own candidates.

    Returns the rows kept, a (query_id, video_id) -> score lookup, and the number
    of skipped rows. A row is skipped when its query is not in `rows` or one of
    its candidates has no cached features.
    """
    text_by_query = {
        str(row["query_id"]): embedding for row, embedding in zip(rows, text_embeddings)
    }
    kept = [
        row
        for row in candidate_rows
        if str(row["query_id"]) in text_by_query
        and all(feature_path(feature_dir, str(v)).exists() for v in row["candidate_video_ids"])
    ]
    skipped = len(candidate_rows) - len(kept)
    if not kept:
        return [], {}, skipped

    video_ids = unique_in_order(str(v) for row in kept for v in row["candidate_video_ids"])
    video_index = {video_id: i for i, video_id in enumerate(video_ids)}
    videos = encode_video_bank(model, video_ids, feature_dir, batch_size, num_workers)

    score_lookup = {}
    for row in kept:
        candidates = row["candidate_video_ids"]
        scores = score_transition_retrieval(
            text_by_query[str(row["query_id"])][None],
            videos[[video_index[str(v)] for v in candidates]],
            model,
        )[0]
        # Keys keep the row's own id values, which is how the evaluator looks them up.
        score_lookup.update(
            {(row["query_id"], v): float(s) for v, s in zip(candidates, scores)}
        )
    return kept, score_lookup, skipped


def evaluate_candidate_file(
    model: TransitionRetrievalModel,
    rows: list[dict],
    text_embeddings: np.ndarray,
    path: Path | None,
    feature_dir: Path,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> tuple[dict | None, int]:
    """Evaluate one hard-negative-schema file; (None, 0) when the file is missing."""
    if path is None or not Path(path).exists():
        return None, 0
    kept, score_lookup, skipped = score_candidate_sets(
        model, rows, text_embeddings, load_jsonl(path), feature_dir, batch_size, num_workers
    )
    if not kept:
        return None, skipped
    return summarize_candidate_set(kept, score_lookup), skipped


def save_transition_caches(
    model: TransitionRetrievalModel,
    video_ids: list[str],
    feature_dir: Path,
    output_dir: Path,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> None:
    """Write shared-format transition caches for the given videos."""
    representations = encode_video_bank(model, video_ids, feature_dir, batch_size, num_workers)
    for video_id, representation in tqdm(
        zip(video_ids, representations), total=len(video_ids), desc="Saving transitions"
    ):
        save_transition_feature_cache(
            video_id, load_states(feature_dir, video_id), representation, output_dir
        )


def seed_checkpoint_path(checkpoint_dir: Path, run_name: str, seed: int) -> Path:
    """Return the best checkpoint of one seeded training run (see train.checkpoint_path)."""
    return Path(checkpoint_dir) / f"{run_name}_seed{seed}_best.pt"


# Main function

def evaluate_transition_model(
    checkpoint_path: Path,
    annotations: Path,
    feature_dir: Path,
    output_path: Path,
    hard_negatives: Path | None = None,
    transition_dir: Path | None = None,
    device: str | None = None,
    limit: int | None = None,
    *,
    action_contrast: Path | None = None,
    model_name: str = MODEL_NAME,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> dict:
    """Evaluate one checkpoint, save the result JSON, and return it."""
    device = get_device(device)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    rows, skipped = load_split_rows(annotations, feature_dir, limit)
    if not rows:
        raise ValueError(f"no annotation rows with cached features in {feature_dir}")
    text_embeddings = encode_row_texts(
        rows, checkpoint["text_encoder"], checkpoint["text_model_name"], device
    )
    summary = checkpoint.get("training_summary") or {}

    result = {
        "model_name": model_name,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "seed": checkpoint.get("seed"),
        "best_epoch": summary.get("best_epoch", checkpoint.get("epoch")),
        "epochs_run": summary.get("epochs_run"),
        "training_summary": summary or None,
        "split": rows[0]["split"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "annotations": str(annotations),
        "feature_dir": str(feature_dir),
        "model_config": checkpoint["model_config"],
        "video_input": model.video_input,
        "query_conditioned": checkpoint["model_config"]["query_conditioned"],
        "text_encoder": checkpoint["text_encoder"],
        "text_model_name": checkpoint["text_model_name"],
        "skipped_missing_features": skipped,
        "full": evaluate_rows(model, rows, text_embeddings, feature_dir, batch_size, num_workers),
    }
    for key, path in (("hard_negative", hard_negatives), ("action_contrast", action_contrast)):
        metrics, set_skipped = evaluate_candidate_file(
            model, rows, text_embeddings, path, feature_dir, batch_size, num_workers
        )
        result[key] = metrics
        result[f"{key}_skipped"] = set_skipped
        result[f"{key}_file"] = str(path) if path is not None else None
    if transition_dir is not None:
        save_transition_caches(
            model, candidate_video_ids(rows), feature_dir, transition_dir, batch_size, num_workers
        )

    save_metrics(result, output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="evaluate this checkpoint only; otherwise evaluate every seed in --seeds",
    )
    parser.add_argument("--run-name", default=MODEL_NAME, help="run name used in training")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(TRAIN_SEEDS))
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--annotations", type=Path, default=ANNOTATION_ROOT / "test.jsonl")
    parser.add_argument("--feature-dir", type=Path, default=VJEPA_FEATURE_ROOT)
    parser.add_argument(
        "--hard-negatives", type=Path, default=ANNOTATION_ROOT / HARD_NEGATIVE_FILE
    )
    parser.add_argument(
        "--action-contrast", type=Path, default=ANNOTATION_ROOT / ACTION_CONTRAST_FILE
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="result file for --checkpoint; seeded runs write <result-dir>/<run>_seed<s>.json",
    )
    parser.add_argument("--result-dir", type=Path, default=RESULT_ROOT)
    parser.add_argument(
        "--model-name", default=None, help="name recorded in the result file (default: run name)"
    )
    parser.add_argument(
        "--transition-dir",
        type=Path,
        default=None,
        help="also save per-video transition caches to this directory",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    model_name = args.model_name or args.run_name
    if args.checkpoint is not None:
        output = args.output or args.result_dir / f"{model_name}_full.json"
        jobs = [(args.checkpoint, output)]
    else:
        jobs = [
            (
                seed_checkpoint_path(args.checkpoint_dir, args.run_name, seed),
                args.result_dir / f"{args.run_name}_seed{seed}.json",
            )
            for seed in args.seeds
        ]
        missing = [str(path) for path, _ in jobs if not path.exists()]
        if missing:
            parser.error(f"missing checkpoints: {missing}")

    for checkpoint, output in jobs:
        result = evaluate_transition_model(
            checkpoint,
            args.annotations,
            args.feature_dir,
            output,
            hard_negatives=args.hard_negatives,
            transition_dir=args.transition_dir,
            device=args.device,
            limit=args.limit,
            action_contrast=args.action_contrast,
            model_name=model_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        summary = {
            key: result[key] and {k: v for k, v in result[key].items() if k != "by_negative_type"}
            for key in ("full", "hard_negative", "action_contrast")
        }
        print(json.dumps({"checkpoint": str(checkpoint), **summary}, indent=2))
        print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
