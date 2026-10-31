"""Evaluate a trained transition model on full and hard-negative retrieval."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ANNOTATION_ROOT, CHECKPOINT_ROOT, FEATURE_ROOT, RESULT_ROOT
from src.eval.evaluate import (
    compute_hard_negative_metrics,
    compute_retrieval_metrics,
    save_metrics,
)
from src.models.transition_model.encoder import encode_text_queries
from src.models.transition_model.retrieval import (
    TransitionRetrievalModel,
    load_checkpoint,
    score_transition_retrieval,
)
from src.models.transition_model.transition import save_transition_feature_cache
from src.models.transition_model.utils import (
    TransitionDataset,
    collate_transitions,
    feature_path,
    get_device,
    load_split_rows,
    load_states,
)


# Config

MODEL_NAME = "transition_model"
BATCH_SIZE = 256
NUM_WORKERS = 0


# Sub functions

def encode_row_texts(
    rows: list[dict],
    encoder_name: str,
    model_name: str,
    device: str | None = None,
) -> np.ndarray:
    """Encode each unique query text once and return one row per annotation."""
    texts = [row["text_query"] for row in rows]
    unique_texts = list(dict.fromkeys(texts))
    embeddings = encode_text_queries(
        unique_texts, encoder_name=encoder_name, model_name=model_name, device=device
    )
    index = {text: position for position, text in enumerate(unique_texts)}
    return embeddings[[index[text] for text in texts]]


def encode_video_bank(
    model: TransitionRetrievalModel,
    video_ids: list[str],
    feature_dir: Path,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> np.ndarray:
    """Return encoded video representations (N, model_dim) in video_ids order."""
    dataset = TransitionDataset(video_ids, list(range(len(video_ids))), feature_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_transitions,
    )
    model.eval()
    device = next(model.parameters()).device
    chunks = []
    with torch.inference_mode():
        for transitions, mask, _ in loader:
            videos = model.encode_videos(transitions.to(device), mask.to(device))
            chunks.append(videos.float().cpu().numpy())
    return np.concatenate(chunks)


def evaluate_rows(
    model: TransitionRetrievalModel,
    rows: list[dict],
    text_embeddings: np.ndarray,
    feature_dir: Path,
) -> dict:
    """Score every query against the split's target videos with the shared evaluator."""
    candidate_ids = list(dict.fromkeys(str(row["target_video_id"]) for row in rows))
    videos = encode_video_bank(model, candidate_ids, feature_dir)
    scores = score_transition_retrieval(text_embeddings, videos, model)
    return compute_retrieval_metrics(
        scores,
        [str(row["query_id"]) for row in rows],
        candidate_ids,
        [str(row["target_video_id"]) for row in rows],
    )


def load_jsonl(path: Path) -> list[dict]:
    """Read one JSON object per nonempty line."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate_hard_negatives(
    model: TransitionRetrievalModel,
    rows: list[dict],
    text_embeddings: np.ndarray,
    hard_negative_path: Path,
    feature_dir: Path,
) -> tuple[dict | None, int]:
    """Score each hard-negative row against its own candidates; return metrics and skipped count."""
    text_by_query = {str(row["query_id"]): embedding for row, embedding in zip(rows, text_embeddings)}
    all_rows = load_jsonl(hard_negative_path)
    hard_rows = [
        row
        for row in all_rows
        if str(row["query_id"]) in text_by_query
        and all(feature_path(feature_dir, str(v)).exists() for v in row["candidate_video_ids"])
    ]
    skipped = len(all_rows) - len(hard_rows)
    if not hard_rows:
        return None, skipped

    video_ids = list(dict.fromkeys(str(v) for row in hard_rows for v in row["candidate_video_ids"]))
    video_index = {video_id: i for i, video_id in enumerate(video_ids)}
    videos = encode_video_bank(model, video_ids, feature_dir)

    score_lookup = {}
    for row in hard_rows:
        query_id = str(row["query_id"])
        candidates = [str(v) for v in row["candidate_video_ids"]]
        scores = score_transition_retrieval(
            text_by_query[query_id][None],
            videos[[video_index[v] for v in candidates]],
            model,
        )[0]
        score_lookup.update({(query_id, v): float(s) for v, s in zip(candidates, scores)})
    return compute_hard_negative_metrics(hard_rows, score_lookup), skipped


def save_transition_caches(
    model: TransitionRetrievalModel,
    video_ids: list[str],
    feature_dir: Path,
    output_dir: Path,
) -> None:
    """Write shared-format transition caches for the given videos."""
    representations = encode_video_bank(model, video_ids, feature_dir)
    for video_id, representation in tqdm(
        zip(video_ids, representations), total=len(video_ids), desc="Saving transitions"
    ):
        save_transition_feature_cache(
            video_id, load_states(feature_dir, video_id), representation, output_dir
        )


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

    result = {
        "model_name": MODEL_NAME,
        "checkpoint_path": str(checkpoint_path),
        "split": rows[0]["split"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "query_conditioned": checkpoint["model_config"]["query_conditioned"],
        "skipped_missing_features": skipped,
        "full": evaluate_rows(model, rows, text_embeddings, feature_dir),
        "hard_negative": None,
    }
    if hard_negatives is not None and Path(hard_negatives).exists():
        metrics, hard_skipped = evaluate_hard_negatives(
            model, rows, text_embeddings, hard_negatives, feature_dir
        )
        result["hard_negative"] = metrics
        result["hard_negative_skipped"] = hard_skipped
    if transition_dir is not None:
        video_ids = list(dict.fromkeys(str(row["target_video_id"]) for row in rows))
        save_transition_caches(model, video_ids, feature_dir, transition_dir)

    save_metrics(result, output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=CHECKPOINT_ROOT / "transition_model_best.pt"
    )
    parser.add_argument("--annotations", type=Path, default=ANNOTATION_ROOT / "test.jsonl")
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_ROOT / "vjepa")
    parser.add_argument(
        "--hard-negatives", type=Path, default=ANNOTATION_ROOT / "hard_negatives.jsonl"
    )
    parser.add_argument(
        "--output", type=Path, default=RESULT_ROOT / "transition_model_full.json"
    )
    parser.add_argument("--transition-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    result = evaluate_transition_model(
        args.checkpoint,
        args.annotations,
        args.feature_dir,
        args.output,
        hard_negatives=args.hard_negatives,
        transition_dir=args.transition_dir,
        device=args.device,
        limit=args.limit,
    )
    print(json.dumps({"full": result["full"], "hard_negative": result["hard_negative"]}, indent=2))
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
