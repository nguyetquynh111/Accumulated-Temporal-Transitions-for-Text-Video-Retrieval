import json
from pathlib import Path
import tempfile
import unittest

from src.models.transition_model.pipelines.report import (
    E1_RUN,
    E2_RUN,
    Results,
    build_report,
    paired_comparison,
)


SEEDS = (42, 43, 44)


def write_run(result_dir: Path, run: str, seed: int, ranks: dict[str, tuple]) -> None:
    """Write per-query ranks and the matching metrics JSON for one run."""
    with (result_dir / f"per_query_{run}_seed{seed}.jsonl").open("w", encoding="utf-8") as handle:
        for query_id, (full, hard, contrast) in ranks.items():
            handle.write(json.dumps({"query_id": query_id, "full_rank": full,
                                     "hard_negative_rank": hard, "action_contrast_rank": contrast}) + "\n")
    result = {"best_epoch": 3}
    for index, setting in enumerate(("full", "hard_negative", "action_contrast")):
        values = [r[index] for r in ranks.values() if r[index] is not None]
        result[setting] = {
            "recall@1": sum(v == 1 for v in values) / len(values),
            "mrr": sum(1 / v for v in values) / len(values),
        }
    (result_dir / f"{run}_seed{seed}.json").write_text(json.dumps(result), encoding="utf-8")


class ReportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.result_dir = Path(self.tmp.name)
        # E2 ranks the action-contrast target first for every query; E1 for half of them.
        e2 = {f"q{i}": (2, 1, 1 if i < 40 else None) for i in range(50)}
        e1 = {f"q{i}": (2, 1, (1 if i % 2 else 2) if i < 40 else None) for i in range(50)}
        for seed in SEEDS:
            write_run(self.result_dir, E2_RUN, seed, e2)
            write_run(self.result_dir, E1_RUN, seed, e1)

    def tearDown(self):
        self.tmp.cleanup()

    def test_paired_comparison(self):
        comparison = paired_comparison(Results([self.result_dir]), seeds=SEEDS, samples=200)
        contrast = comparison["settings"]["action_contrast"]
        self.assertEqual(contrast["num_queries"], 40)
        self.assertAlmostEqual(contrast["recall@1"]["difference"], 50.0)
        self.assertTrue(contrast["recall@1"]["criterion_met"])
        for seed in SEEDS:
            bootstrap = contrast["recall@1"]["bootstrap"][str(seed)]
            self.assertAlmostEqual(bootstrap["difference"], 50.0)
            self.assertGreater(bootstrap["ci95"][0], 0)
        full = comparison["settings"]["full"]["recall@1"]
        self.assertEqual(full["difference"], 0.0)
        self.assertFalse(full["criterion_met"])

    def test_later_folder_replaces_earlier(self):
        later = self.result_dir / "later"
        later.mkdir()
        (later / f"{E2_RUN}_seed42.json").write_text(json.dumps({"full": {"mrr": 0.5}}), encoding="utf-8")
        runs = Results([self.result_dir, later]).runs(E2_RUN)
        self.assertEqual([r["full"]["mrr"] for r in runs][0], 0.5)
        self.assertEqual(len(runs), 3)

    def test_build_report_writes_tables(self):
        output_dir = self.result_dir / "generated"
        build_report(self.result_dir, output_dir, samples=200)
        for name in ("numbers.tex", "tab_main.tex", "tab_paired.tex", "tab_bytype.tex",
                     "tab_ablation.tex", "tab_reverse.tex"):
            self.assertTrue((output_dir / name).exists(), name)
        self.assertFalse((output_dir / "sec_errors.tex").exists())
        numbers = (output_dir / "numbers.tex").read_text(encoding="utf-8")
        self.assertIn(r"\newcommand{\DeltaAcRone}{+50.00}", numbers)
        self.assertIn(r"\newcommand{\DeltaAcRoneLowFortytwo}", numbers)
        self.assertTrue((self.result_dir / "e1_e2_paired_bootstrap.json").exists())

    def test_saved_metrics_must_match(self):
        path = self.result_dir / f"{E2_RUN}_seed42.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        result["action_contrast"]["recall@1"] = 0.1
        path.write_text(json.dumps(result), encoding="utf-8")
        with self.assertRaises(ValueError):
            build_report(self.result_dir, self.result_dir / "generated", samples=50)


if __name__ == "__main__":
    unittest.main()
