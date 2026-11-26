import csv
import json

import pytest

from src.eval.evaluate import (
    build_comparison_table,
    paired_bootstrap_recall_at_1,
)


def test_build_comparison_table(tmp_path):
    e2_name = "two_stage_token_reranker"
    e1_name = "two_stage_token_reranker_state_only"

    model_results = {
        e2_name: [0.70, 0.75, 0.80],
        e1_name: [0.60, 0.65, 0.70],
    }

    result_paths = []

    for model, scores in model_results.items():
        for seed, score in zip((42, 43, 44), scores):
            result = {
                "model_name": model,
                "variant": model,
                "split": "test",
                "seed": seed,
                "full": {
                    "recall@1": score,
                    "num_queries": 4,
                    "num_candidates": 10,
                },
                "hard_negative": {
                    "recall@1": score,
                    "num_queries": 4,
                    "num_candidates": 10,
                },
                "action_contrast": {
                    "recall@1": score,
                    "num_queries": 4,
                    "num_candidates": 10,
                },
            }

            path = tmp_path / f"{model}_seed{seed}.json"
            path.write_text(json.dumps(result), encoding="utf-8")
            result_paths.append(path)

    output_path = tmp_path / "comparison_table.csv"

    saved_path = build_comparison_table(result_paths, output_path)

    assert saved_path == output_path
    assert output_path.exists()

    with output_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 2

    by_model = {row["model"]: row for row in rows}

    e2 = by_model[e2_name]
    e1 = by_model[e1_name]

    assert e2["seeds"] == "42,43,44"
    assert e2["num_runs"] == "3"

    assert float(e2["action_contrast_recall@1_mean"]) == 0.75
    assert float(e1["action_contrast_recall@1_mean"]) == 0.65

    assert float(e2["action_contrast_recall@1_std"]) == 0.05

    for seed in (42, 43, 44):
        column = f"e2_minus_e1_action_contrast_recall@1_pp_seed{seed}"
        assert float(e2[column]) == 10.0

    assert e2["full_mrr_mean"] == "N/A"


def test_paired_bootstrap_recall_at_1():

    e1_rows = [
        {"query_id": "q1", "action_contrast_rank": 2},
        {"query_id": "q2", "action_contrast_rank": 1},
        {"query_id": "q3", "action_contrast_rank": 3},
        {"query_id": "q4", "action_contrast_rank": 1},
    ]

    e2_rows = [
        {"query_id": "q1", "action_contrast_rank": 1},
        {"query_id": "q2", "action_contrast_rank": 1},
        {"query_id": "q3", "action_contrast_rank": 1},
        {"query_id": "q4", "action_contrast_rank": 1},
    ]

    result = paired_bootstrap_recall_at_1(
        e1_rows,
        e2_rows,
        n_bootstrap=1000,
        seed=42,
    )

    assert result["difference_pp"] == 50.0
    assert result["num_queries"] == 4
    assert result["n_bootstrap"] == 1000

    assert result["ci_low_pp"] <= 50.0
    assert result["ci_high_pp"] >= 50.0

    repeated_result = paired_bootstrap_recall_at_1(
        e1_rows,
        e2_rows,
        n_bootstrap=1000,
        seed=42,
    )

    assert result == repeated_result


def test_paired_bootstrap_invalid_data():
    e1_rows = [
        {"query_id": "q1", "action_contrast_rank": 2},
        {"query_id": "q2", "action_contrast_rank": 1},
    ]

    e2_wrong_queries = [
        {"query_id": "q1", "action_contrast_rank": 1},
        {"query_id": "q3", "action_contrast_rank": 1},
    ]

    with pytest.raises(ValueError, match="same query IDs"):
        paired_bootstrap_recall_at_1(e1_rows, e2_wrong_queries)

    e2_invalid_rank = [
        {"query_id": "q1", "action_contrast_rank": 0},
        {"query_id": "q2", "action_contrast_rank": 1},
    ]

    with pytest.raises(ValueError, match="Invalid rank"):
        paired_bootstrap_recall_at_1(e1_rows, e2_invalid_rank)

    e2_missing_rank = [
        {"query_id": "q1", "action_contrast_rank": None},
        {"query_id": "q2", "action_contrast_rank": 1},
    ]

    with pytest.raises(ValueError, match="Missing matched rank"):
        paired_bootstrap_recall_at_1(e1_rows, e2_missing_rank)


def test_comparison_table_with_bootstrap_files(tmp_path):
    e1_name = "two_stage_token_reranker_state_only"
    e2_name = "two_stage_token_reranker"

    model_data = {
        e1_name: [2, 1, 2, 1],
        e2_name: [1, 1, 1, 1],
    }

    result_paths = []

    for model, ranks in model_data.items():
        recall_at_1 = sum(rank == 1 for rank in ranks) / len(ranks)

        result = {
            "model_name": model,
            "variant": model,
            "split": "test",
            "seed": 42,
            "full": {
                "recall@1": recall_at_1,
                "num_queries": 4,
                "num_candidates": 10,
            },
            "action_contrast": {
                "recall@1": recall_at_1,
                "num_queries": 4,
                "num_candidates": 4,
            },
        }

        result_path = tmp_path / f"{model}_seed42.json"
        result_path.write_text(json.dumps(result), encoding="utf-8")

        result_paths.append(result_path)

        per_query_path = tmp_path / f"per_query_{model}_seed42.jsonl"

        with per_query_path.open("w", encoding="utf-8") as file:
            for index, rank in enumerate(ranks, start=1):
                row = {
                    "query_id": f"q{index}",
                    "action_contrast_rank": rank,
                }

                file.write(json.dumps(row) + "\n")

    output_path = tmp_path / "comparison_table.csv"

    build_comparison_table(result_paths, output_path)

    with output_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    by_model = {row["model"]: row for row in rows}

    e2 = by_model[e2_name]

    difference_column = "e2_minus_e1_action_contrast_recall@1_pp_seed42"

    low_column = "e2_minus_e1_action_contrast_recall@1_ci_low_pp_seed42"
    high_column = "e2_minus_e1_action_contrast_recall@1_ci_high_pp_seed42"

    count_column = "e2_minus_e1_action_contrast_num_queries_seed42"

    assert float(e2[difference_column]) == 50.0

    assert e2[low_column] != "N/A"
    assert e2[high_column] != "N/A"

    assert float(e2[low_column]) <= 50.0
    assert float(e2[high_column]) >= 50.0

    assert int(e2[count_column]) == 4