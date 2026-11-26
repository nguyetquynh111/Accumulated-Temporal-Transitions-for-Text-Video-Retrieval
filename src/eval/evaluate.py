import random
import re
from collections import defaultdict

from config import RANDOM_SEED

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


# SSv2 label pairs treated as opposite actions.
OPPOSITE_LABEL_PAIRS = (
    ("closing something", "opening something"),
    ("covering something with something", "uncovering something"),
    ("folding something", "unfolding something"),
    ("putting something into something", "taking something out of something"),
    ("putting something into something", "pulling something out of something"),
    ("stuffing something into something", "taking something out of something"),
    ("stuffing something into something", "pulling something out of something"),
    ("burying something in something", "digging something out of something"),
    ("picking something up", "putting something on a surface"),
    (
        "putting something similar to other things that are already on the table",
        "taking one of many similar things on the table",
    ),
    ("pouring something into something", "pouring something out of something"),
    ("pushing something onto something", "pushing something off of something"),
    ("moving something up", "moving something down"),
    ("moving something closer to something", "moving something away from something"),
    (
        "moving something and something closer to each other",
        "moving something and something away from each other",
    ),
    ("moving something towards the camera", "moving something away from the camera"),
    ("approaching something with your camera", "moving away from something with your camera"),
    ("pushing something from left to right", "pushing something from right to left"),
    ("pulling something from left to right", "pulling something from right to left"),
    ("turning the camera left while filming something", "turning the camera right while filming something"),
    (
        "turning the camera upwards while filming something",
        "turning the camera downwards while filming something",
    ),
)


def _normalize_label(value: Any) -> str:
    return " ".join(str(value).lower().replace("[", "").replace("]", "").split())


OPPOSITE_LABELS: dict[str, frozenset[str]] = {
    label: frozenset(
        _normalize_label(other)
        for a, b in OPPOSITE_LABEL_PAIRS
        for this, other in ((a, b), (b, a))
        if _normalize_label(this) == label
    )
    for label in {_normalize_label(x) for pair in OPPOSITE_LABEL_PAIRS for x in pair}
}


def is_opposite_action(label_a: str, label_b: str) -> bool:
    """Return whether two action labels are a hand-checked opposite pair."""
    return _normalize_label(label_b) in OPPOSITE_LABELS.get(_normalize_label(label_a), ())


def build_hard_negative_set(
    annotations: list[dict],
    output_path: Path,
) -> Path:
    """Build hard-negative candidate sets from one evaluation split."""

    if not annotations:
        raise ValueError("annotations must not be empty")

    # Use only one evaluation split to prevent data leakage.
    splits = {str(row.get("split", "")) for row in annotations}

    if len(splits) != 1 or not splits.issubset({"val", "test"}):
        raise ValueError(
            "Pass annotations from one split: val OR test"
        )

    rows = sorted(
        annotations,
        key=lambda row: str(row["query_id"])
    )

    required = {
        "query_id",
        "target_video_id",
        "video_id",
        "text_query",
    }

    for row in rows:
        if not required.issubset(row):
            raise ValueError(
                f"Missing required fields: {required - set(row)}"
            )

        if str(row["video_id"]) != str(row["target_video_id"]):
            raise ValueError(
                "Each source video must match its target_video_id"
            )

    video_by_id = {
        str(row["video_id"]): row
        for row in rows
    }

    if len(video_by_id) != len(rows):
        raise ValueError(
            "Expected one annotation per video_id"
        )

    if len({
        str(row["query_id"]) for row in rows
    }) != len(rows):
        raise ValueError("query_id values must be unique")

    def normalized(value):
        return " ".join(
            str(value)
            .lower()
            .replace("[", "")
            .replace("]", "")
            .split()
        )

    def opposite(a, b):
        return is_opposite_action(a, b)

    def objects(row):
        values = row.get("objects") or []

        if not isinstance(values, list):
            return set()

        return {
            normalized(value)
            for value in values
            if normalized(value) not in {
                "",
                "something",
                "anything",
                "object",
                "item",
            }
        }
    def primary_object(row):
        # Use the first annotated object, ignoring generic placeholders.
        values = row.get("objects") or []

        if not isinstance(values, list) or not values:
            return ""

        primary = normalized(values[0])

        generic_objects = {
            "",
            "something",
            "anything",
            "object",
            "item",
            "stuff",
        }

        if primary in generic_objects:
            return ""

        return primary

    labels = {
        vid: normalized(
            row.get("label")
            or row.get("template")
            or row["text_query"]
        )
        for vid, row in video_by_id.items()
    }

    texts = {
        vid: normalized(row["text_query"])
        for vid, row in video_by_id.items()
    }

    by_label = defaultdict(list)
    by_object = defaultdict(list)

    for video_id in sorted(video_by_id):
        by_label[labels[video_id]].append(video_id)

        for obj in sorted(objects(video_by_id[video_id])):
            by_object[obj].append(video_id)

    # Find the action labels that are potential opposites.
    opposite_labels = {
        label: [
            other
            for other in sorted(by_label)
            if other != label and opposite(label, other)
        ]
        for label in by_label
    }

    all_video_ids = sorted(video_by_id)

    rng = random.Random(RANDOM_SEED)

    output_rows = []

    for row in rows:
        target_id = str(row["target_video_id"])
        label = labels[target_id]

        opposite_ids = list(dict.fromkeys(
            vid
            for action in opposite_labels[label]
            for vid in by_label[action]
        ))

        shared_ids = list(dict.fromkeys(
            vid
            for obj in sorted(objects(row))
            for vid in by_object[obj]
        ))

        negatives = []
        negative_types = []
        seen = {target_id}

        def add(pool, kind):
            if not pool or len(negatives) == 9:
                return

            start = rng.randrange(len(pool))

            for i in range(len(pool)):
                vid = pool[(start + i) % len(pool)]

                # Do not use the target or an identical query.
                if (
                    vid in seen
                    or texts[vid] == texts[target_id]
                    or labels[vid] == label
                ):
                    continue

                negatives.append(vid)
                negative_types.append(kind)
                seen.add(vid)

                if len(negatives) == 9:
                    break

        # Priority 1: Opposite actions involving the same primary object.
        primary = primary_object(row)

        matching_opposites = [
            vid for vid in opposite_ids
            if primary
            and primary_object(video_by_id[vid]) == primary
        ]

        add(matching_opposites, "opposite_action")

        # Priority 2: Different actions involving shared objects.
        add(shared_ids, "same_objects")

        # Priority 3: Remaining different-action videos.
        add(all_video_ids, "other_action")

        if len(negatives) < 2:
            raise ValueError(
                f"Not enough distinct negative videos for query "
                f"{row['query_id']}"
            )

        # Vary the correct video's position to avoid ranking bias.
        candidates = negatives.copy()

        candidates.insert(
            rng.randrange(len(candidates) + 1),
            target_id
        )

        output_rows.append({
            "query_id": str(row["query_id"]),
            "target_video_id": target_id,
            "candidate_video_ids": candidates,
            "negative_types": negative_types,
        })

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open("w", encoding="utf-8") as handle:
        for item in output_rows:
            handle.write(
                json.dumps(item, ensure_ascii=False) + "\n"
            )

    return output_path

def build_comparison_table(
    result_paths: list[Path],
    output_path: Path,
) -> Path:
    """Create a CSV comparison table from saved model evaluation results."""

    import csv
    from statistics import mean, stdev

    if not result_paths:
        raise ValueError("result_paths must not be empty")

    metric_names = ("recall@1", "recall@5", "recall@10", "mrr", "map")
    settings = ("full", "hard_negative", "action_contrast")
    negative_types = ("opposite_action", "same_objects", "other_action")

    runs_by_model = defaultdict(list)
    signatures = {}

    for path in result_paths:
        path = Path(path)

        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)

        if not isinstance(result, dict) or not isinstance(result.get("full"), dict):
            raise ValueError(f"Missing full retrieval results in {path}")

        split = result.get("split")

        if not split:
            raise ValueError(f"Missing split name in {path}")

        model = str(result.get("variant") or result.get("model_name") or path.stem)

        if result.get("reverse_time") or path.stem.endswith("_reversed"):
            model += "_reversed"

        for setting in settings:
            section = result.get(setting)

            if section is None:
                continue

            if not isinstance(section, dict):
                raise ValueError(f"Invalid {setting} results in {path}")

            counts = (
                split,
                section.get("num_queries"),
                section.get("num_candidates"),
            )

            if setting in signatures and signatures[setting] != counts:
                raise ValueError(
                    f"Different split or candidate counts for {setting}: {path}"
                )

            signatures.setdefault(setting, counts)

        existing = runs_by_model[(model, split)]

        if any(run.get("seed") == result.get("seed") for run in existing):
            raise ValueError(f"Duplicate seed for {model}: {result.get('seed')}")

        result["_per_query_path"] = path.with_name(
            f"per_query_{path.stem}.jsonl"
        )

        existing.append(result)

    columns = ["model", "split", "seeds", "num_runs"]

    for setting in settings:
        columns.append(f"{setting}_num_queries")

        for metric in metric_names:
            columns.extend([
                f"{setting}_{metric}_mean",
                f"{setting}_{metric}_std",
            ])

    for kind in negative_types:
        for metric in ("recall@1", "mrr", "pairwise_accuracy"):
            columns.extend([
                f"hard_negative_{kind}_{metric}_mean",
                f"hard_negative_{kind}_{metric}_std",
            ])

    columns.extend([
        "action_contrast_opposite_action_pairwise_accuracy_mean",
        "action_contrast_opposite_action_pairwise_accuracy_std",
    ])

    def summary(runs, setting, metric, kind=None):
        numbers = []

        for run in runs:
            section = run.get(setting)

            if kind is not None and isinstance(section, dict):
                section = section.get("by_negative_type", {}).get(kind)

            value = section.get(metric) if isinstance(section, dict) else None

            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return "N/A", "N/A"

            numbers.append(float(value))

        average = round(mean(numbers), 6)

        standard_deviation = (
            round(stdev(numbers), 6) if len(numbers) > 1 else "N/A"
        )

        return average, standard_deviation

    rows = []

    for (model, split), runs in sorted(runs_by_model.items()):
        seed_names = sorted(
            str(run["seed"])
            for run in runs
            if run.get("seed") is not None
        )

        row = {
            "model": model,
            "split": split,
            "seeds": ",".join(seed_names) if seed_names else "N/A",
            "num_runs": len(runs),
        }

        for setting in settings:
            section = next(
                (
                    run.get(setting)
                    for run in runs
                    if isinstance(run.get(setting), dict)
                ),
                None,
            )

            row[f"{setting}_num_queries"] = (
                section.get("num_queries", "N/A") if section else "N/A"
            )

            for metric in metric_names:
                avg, std = summary(runs, setting, metric)

                row[f"{setting}_{metric}_mean"] = avg
                row[f"{setting}_{metric}_std"] = std

        for kind in negative_types:
            for metric in ("recall@1", "mrr", "pairwise_accuracy"):
                avg, std = summary(runs, "hard_negative", metric, kind)

                row[f"hard_negative_{kind}_{metric}_mean"] = avg
                row[f"hard_negative_{kind}_{metric}_std"] = std

        avg, std = summary(
            runs,
            "action_contrast",
            "pairwise_accuracy",
            "opposite_action",
        )

        row["action_contrast_opposite_action_pairwise_accuracy_mean"] = avg
        row["action_contrast_opposite_action_pairwise_accuracy_std"] = std

        rows.append(row)

    comparison_columns = [
        f"e2_minus_e1_{setting}_recall@1_pp_seed{seed}"
        for setting in settings
        for seed in (42, 43, 44)
    ]

    bootstrap_columns = [
        f"e2_minus_e1_action_contrast_recall@1_ci_{bound}_pp_seed{seed}"
        for seed in (42, 43, 44)
        for bound in ("low", "high")
    ]

    bootstrap_columns.extend([
        f"e2_minus_e1_action_contrast_num_queries_seed{seed}"
        for seed in (42, 43, 44)
    ])

    columns.extend(comparison_columns + bootstrap_columns)

    for row in rows:
        for column in comparison_columns + bootstrap_columns:
            row[column] = "N/A"

    e2_name = "two_stage_token_reranker"
    e1_name = "two_stage_token_reranker_state_only"

    for (model, split), e2_runs in runs_by_model.items():
        if model != e2_name:
            continue

        e1_runs = runs_by_model.get((e1_name, split))

        if not e1_runs:
            continue

        e2_by_seed = {
            str(run["seed"]): run
            for run in e2_runs
            if run.get("seed") is not None
        }

        e1_by_seed = {
            str(run["seed"]): run
            for run in e1_runs
            if run.get("seed") is not None
        }

        e2_row = next(
            row for row in rows
            if row["model"] == e2_name and row["split"] == split
        )

        for setting in settings:
            for seed in (42, 43, 44):
                e2_run = e2_by_seed.get(str(seed))
                e1_run = e1_by_seed.get(str(seed))

                if e2_run is None or e1_run is None:
                    continue

                e2_metrics = e2_run.get(setting)
                e1_metrics = e1_run.get(setting)

                if not isinstance(e2_metrics, dict) or not isinstance(e1_metrics, dict):
                    continue

                e2_value = e2_metrics.get("recall@1")
                e1_value = e1_metrics.get("recall@1")

                if not isinstance(e2_value, (int, float)):
                    continue

                if not isinstance(e1_value, (int, float)):
                    continue

                difference_pp = 100 * (e2_value - e1_value)

                column = f"e2_minus_e1_{setting}_recall@1_pp_seed{seed}"

                e2_row[column] = round(difference_pp, 3)
                if setting == "action_contrast":
                    e1_path = e1_run["_per_query_path"]
                    e2_path = e2_run["_per_query_path"]

                    if e1_path.is_file() and e2_path.is_file():
                        with e1_path.open("r", encoding="utf-8") as file:
                            e1_rows = [
                                json.loads(line)
                                for line in file
                                if line.strip()
                            ]

                        with e2_path.open("r", encoding="utf-8") as file:
                            e2_rows = [
                                json.loads(line)
                                for line in file
                                if line.strip()
                            ]

                        ci = paired_bootstrap_recall_at_1(
                            e1_rows,
                            e2_rows,
                            rank_key="action_contrast_rank",
                            n_bootstrap=10000,
                            seed=42,
                        )

                        prefix = (
                            "e2_minus_e1_action_contrast_recall@1_ci"
                        )

                        e2_row[f"{prefix}_low_pp_seed{seed}"] = ci["ci_low_pp"]
                        e2_row[f"{prefix}_high_pp_seed{seed}"] = ci["ci_high_pp"]

                        e2_row[
                            f"e2_minus_e1_action_contrast_num_queries_seed{seed}"
                        ] = ci["num_queries"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    return output_path
def paired_bootstrap_recall_at_1(
    e1_rows: list[dict],
    e2_rows: list[dict],
    rank_key: str = "action_contrast_rank",
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """Calculate the paired-bootstrap 95% CI for E2 minus E1 Recall@1."""

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")

    def get_ranks(rows):
        ranks = {}

        for row in rows:
            query_id = str(row["query_id"])

            if query_id in ranks:
                raise ValueError(f"Duplicate query_id: {query_id}")

            ranks[query_id] = row.get(rank_key)

        return ranks

    e1_ranks = get_ranks(e1_rows)
    e2_ranks = get_ranks(e2_rows)

    if e1_ranks.keys() != e2_ranks.keys():
        raise ValueError("E1 and E2 must contain the same query IDs")

    differences = []

    for query_id in sorted(e1_ranks):
        e1_rank = e1_ranks[query_id]
        e2_rank = e2_ranks[query_id]

        if e1_rank is None and e2_rank is None:
            continue

        if e1_rank is None or e2_rank is None:
            raise ValueError(f"Missing matched rank for {query_id}")

        if (
            not isinstance(e1_rank, int)
            or not isinstance(e2_rank, int)
            or isinstance(e1_rank, bool)
            or isinstance(e2_rank, bool)
            or e1_rank < 1
            or e2_rank < 1
        ):
            raise ValueError(f"Invalid rank for {query_id}")

        e1_correct = int(e1_rank == 1)
        e2_correct = int(e2_rank == 1)

        differences.append(e2_correct - e1_correct)

    if not differences:
        raise ValueError("No valid paired queries found")

    differences = np.asarray(differences, dtype=np.float64)

    observed_difference = float(np.mean(differences))

    rng = np.random.default_rng(seed)
    bootstrap_differences = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        indices = rng.integers(
            0,
            len(differences),
            size=len(differences),
        )

        bootstrap_differences[i] = np.mean(differences[indices])

    ci_low, ci_high = np.percentile(
        bootstrap_differences,
        [2.5, 97.5],
    )

    return {
        "difference_pp": round(100 * observed_difference, 4),
        "ci_low_pp": round(100 * float(ci_low), 4),
        "ci_high_pp": round(100 * float(ci_high), 4),
        "num_queries": len(differences),
        "n_bootstrap": n_bootstrap,
    }
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
