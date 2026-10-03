import json
from pathlib import Path
from typing import Any

import numpy as np


def compute_retrieval_metrics(
    scores: np.ndarray,
    query_ids: list[str],
    candidate_video_ids: list[str],
    target_video_ids: list[str],
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float | int]:
    """Return Recall@K, MRR, mAP, and count metadata."""
    scores = np.asarray(scores, dtype=np.float32)
    _validate_retrieval_inputs(scores, query_ids, candidate_video_ids, target_video_ids)

    candidate_to_index = {video_id: index for index, video_id in enumerate(candidate_video_ids)}
    ranks = []
    for row, target_video_id in zip(scores, target_video_ids, strict=True):
        if target_video_id not in candidate_to_index:
            raise ValueError(f"target_video_id missing from candidates: {target_video_id}")
        target_index = candidate_to_index[target_video_id]
        order = np.argsort(-row, kind="stable")
        rank = int(np.where(order == target_index)[0][0]) + 1
        ranks.append(rank)

    ranks_array = np.asarray(ranks, dtype=np.float32)
    metrics: dict[str, float | int] = {}
    for k in k_values:
        metrics[f"recall@{k}"] = float(np.mean(ranks_array <= k))
    metrics["mrr"] = float(np.mean(1.0 / ranks_array))
    metrics["map"] = metrics["mrr"]
    metrics["num_queries"] = len(query_ids)
    metrics["num_candidates"] = len(candidate_video_ids)
    return metrics


def compute_hard_negative_metrics(
    hard_negative_rows: list[dict[str, Any]],
    score_lookup: dict[tuple[str, str], float],
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float | int]:
    """Evaluate per-query hard-negative candidate sets."""
    ranks = []
    candidate_counts = []
    for row in hard_negative_rows:
        query_id = row["query_id"]
        target_video_id = row["target_video_id"]
        candidate_video_ids = row["candidate_video_ids"]
        if target_video_id not in candidate_video_ids:
            raise ValueError(f"target_video_id missing from hard negatives: {query_id}")

        scores = np.asarray(
            [score_lookup[(query_id, video_id)] for video_id in candidate_video_ids],
            dtype=np.float32,
        )
        target_index = candidate_video_ids.index(target_video_id)
        order = np.argsort(-scores, kind="stable")
        rank = int(np.where(order == target_index)[0][0]) + 1
        ranks.append(rank)
        candidate_counts.append(len(candidate_video_ids))

    if not ranks:
        raise ValueError("hard_negative_rows must not be empty")

    ranks_array = np.asarray(ranks, dtype=np.float32)
    metrics: dict[str, float | int] = {}
    for k in k_values:
        metrics[f"recall@{k}"] = float(np.mean(ranks_array <= k))
    metrics["mrr"] = float(np.mean(1.0 / ranks_array))
    metrics["map"] = metrics["mrr"]
    metrics["num_queries"] = len(hard_negative_rows)
    metrics["num_candidates"] = int(max(candidate_counts))
    return metrics


def save_metrics(metrics: dict, output_path: Path) -> None:
    """Save metrics as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_hard_negative_set(
    annotations: list[dict],
    output_path: Path,
) -> Path:
    """Create a hard-negative JSONL file and return its path."""
    raise NotImplementedError(
        "Hard-negative construction needs dataset-specific label/object heuristics."
    )


def build_comparison_table(
    result_paths: list[Path],
    output_path: Path,
) -> Path:
    """Create a CSV table comparing all model metrics."""
    raise NotImplementedError("Comparison-table formatting will be added with result files.")


def categorize_retrieval_errors(
    ranked_results: list[dict],
    annotations: dict[str, dict],
) -> list[dict]:
    """Return categorized retrieval failures for qualitative analysis."""
    raise NotImplementedError(
        "Error categories require qualitative rules agreed on after first model runs."
    )


def _validate_retrieval_inputs(
    scores: np.ndarray,
    query_ids: list[str],
    candidate_video_ids: list[str],
    target_video_ids: list[str],
) -> None:
    if scores.ndim != 2:
        raise ValueError("scores must have shape (N_queries, N_candidates)")
    if scores.shape != (len(query_ids), len(candidate_video_ids)):
        raise ValueError("scores shape must match query and candidate ids")
    if len(query_ids) != len(target_video_ids):
        raise ValueError("query_ids and target_video_ids must have the same length")
