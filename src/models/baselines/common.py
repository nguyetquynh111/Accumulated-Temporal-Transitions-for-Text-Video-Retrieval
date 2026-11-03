"""Shared helpers for baseline and ablation runners: rankings, hard negatives, result files."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from config import PROJECT_ROOT
from src.eval.evaluate import compute_hard_negative_metrics, compute_retrieval_metrics


# Config

TOP_K = 10


# Sub functions

def query_ids(rows: list[dict]) -> list[str]:
    return [str(row["query_id"]) for row in rows]


def target_ids(rows: list[dict]) -> list[str]:
    return [str(row["target_video_id"]) for row in rows]


def candidate_ids(rows: list[dict]) -> list[str]:
    """Return the split's unique target videos, in first-seen order."""
    return list(dict.fromkeys(target_ids(rows)))


def resolve_video_row(row: dict) -> dict:
    """Return a copy of row whose video_path is absolute."""
    resolved = dict(row)
    video_path = Path(resolved["video_path"])
    if not video_path.is_absolute():
        resolved["video_path"] = str(PROJECT_ROOT / video_path)
    return resolved


def encode_unique_texts(
    rows: list[dict],
    encode_fn: Callable[[list[str]], np.ndarray],
) -> np.ndarray:
    """Encode each unique query text once and return one row per annotation."""
    texts = [row["text_query"] for row in rows]
    unique_texts = list(dict.fromkeys(texts))
    embeddings = encode_fn(unique_texts)
    index = {text: position for position, text in enumerate(unique_texts)}
    return embeddings[[index[text] for text in texts]]


def load_jsonl(path: Path) -> list[dict]:
    """Read one JSON object per nonempty line."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_jsonl(records: list[dict], path: Path) -> Path:
    """Write one JSON object per line and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def read_hard_negative_rows(path: Path | None, rows: list[dict]) -> list[dict]:
    """Return hard-negative rows whose query belongs to rows; [] if path is missing."""
    if path is None or not Path(path).exists():
        return []
    split_queries = set(query_ids(rows))
    return [row for row in load_jsonl(path) if str(row["query_id"]) in split_queries]


def hard_negative_video_ids(hard_rows: list[dict]) -> list[str]:
    return list(dict.fromkeys(str(v) for row in hard_rows for v in row["candidate_video_ids"]))


def split_hard_negative_rows(
    hard_rows: list[dict],
    available_video_ids: set[str],
) -> tuple[list[dict], int]:
    """Keep rows whose candidates all have features; return them and the skipped count."""
    kept = [
        row
        for row in hard_rows
        if all(str(v) in available_video_ids for v in row["candidate_video_ids"])
    ]
    return kept, len(hard_rows) - len(kept)


# Main functions

def evaluate_full(scores: np.ndarray, rows: list[dict], candidates: list[str]) -> dict:
    """Run the shared evaluator on a full-split score matrix."""
    return compute_retrieval_metrics(scores, query_ids(rows), candidates, target_ids(rows))


def evaluate_hard_negatives(
    hard_rows: list[dict],
    rows: list[dict],
    text_embeddings: np.ndarray,
    video_ids: list[str],
    video_embeddings: np.ndarray,
    score_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> dict:
    """Score each hard-negative row against only its own candidates.

    score_fn maps text embeddings (1, D_text) and candidate embeddings (n, D)
    to scores (1, n), so every model can reuse this with its own scorer.
    """
    text_by_query = dict(zip(query_ids(rows), text_embeddings))
    video_index = {video_id: i for i, video_id in enumerate(video_ids)}
    score_lookup = {}
    for row in hard_rows:
        query_id = str(row["query_id"])
        candidates = [str(v) for v in row["candidate_video_ids"]]
        scores = score_fn(
            text_by_query[query_id][None],
            video_embeddings[[video_index[v] for v in candidates]],
        )[0]
        score_lookup.update({(query_id, v): float(s) for v, s in zip(candidates, scores)})
    return compute_hard_negative_metrics(hard_rows, score_lookup)


def ranking_records(
    model_name: str,
    rows: list[dict],
    scores: np.ndarray,
    candidates: list[str],
    top_k: int = TOP_K,
) -> list[dict]:
    """Return one record per query with the target rank and top-k candidates."""
    candidate_index = {video_id: i for i, video_id in enumerate(candidates)}
    records = []
    for row, row_scores in zip(rows, scores):
        order = np.argsort(-row_scores, kind="stable")
        target = str(row["target_video_id"])
        rank = int(np.where(order == candidate_index[target])[0][0]) + 1
        records.append(
            {
                "model_name": model_name,
                "query_id": str(row["query_id"]),
                "text_query": row["text_query"],
                "target_video_id": target,
                "target_rank": rank,
                "num_candidates": len(candidates),
                "top_video_ids": [candidates[i] for i in order[:top_k]],
                "top_scores": [float(row_scores[i]) for i in order[:top_k]],
            }
        )
    return records


def build_result(model_name: str, rows: list[dict], full: dict, **extra) -> dict:
    """Return a result dict with the fields every result JSON should carry."""
    return {
        "model_name": model_name,
        "split": rows[0]["split"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "full": full,
        "hard_negative": None,
        **extra,
    }
