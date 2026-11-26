"""Build the paper's result tables and numbers from saved result files (T4.3).

Reads the metrics JSON files and per-query ranks in one or more --result-dir
folders (a file in a later folder replaces the same file in an earlier one) and
writes to --output-dir (default paper/generated/):

- numbers.tex: \\newcommand macros for every number quoted in the paper text;
- tab_main.tex: full, hard-negative, and action-contrast retrieval of the
  baselines and of E1 and E2;
- tab_paired.tex: E2 - E1 Recall@1 per seed with its paired-bootstrap 95% CI;
- tab_bytype.tex: hard-negative pairwise accuracy by negative type;
- tab_ablation.tex: every trained variant that has results;
- tab_reverse.tex: sequence models evaluated with states reversed in time;
- sec_errors.tex and tab_errors.tex, only if error_analysis.json exists (T4.1).

The E1 vs E2 confidence intervals come from the shared evaluator
(src.eval.evaluate.paired_bootstrap_recall_at_1, T3.2): one paired bootstrap per
seed, resampling queries paired by query_id. The success criterion of PRD
Section 7 holds when every seed's CI excludes zero on the same side. The result
is also written to e1_e2_paired_bootstrap.json in the first --result-dir.
Nothing is recomputed from features, and the metrics recomputed from the
per-query ranks are checked against the saved JSON.

<<<<<<< HEAD
    python -m src.models.transition_model.pipelines.report
=======
    python -m src.models.transition_model.pipelines.report \\
        --result-dir outputs/results outputs/pr6_t33_ablations/results
>>>>>>> 7db566adf0a83061c85918e3a11ffd5cb7f10719
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from config import PROJECT_ROOT, RESULT_ROOT, TRAIN_SEEDS
from src.eval.evaluate import paired_bootstrap_recall_at_1


# Config

E2_RUN = "two_stage_token_reranker"
E1_RUN = "two_stage_token_reranker_state_only"
SETTINGS = ("full", "hard_negative", "action_contrast")
SETTING_NAMES = {"full": "Full", "hard_negative": "Hard neg.", "action_contrast": "Action contrast"}
NEGATIVE_TYPES = ("opposite_action", "same_objects", "other_action")
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 42
CHECK_TOLERANCE = 1e-4
SEED_WORDS = {42: "Fortytwo", 43: "Fortythree", 44: "Fortyfour"}

BASELINES = (
    ("clip_baseline", "B1: CLIP frame averaging"),
    ("vjepa_baseline", "B2: V-JEPA~2, pooled"),
)
STAGE1_E2 = "ablation_full_transition_without_query_conditioning"
STAGE1_E1 = "ablation_state_only_without_query_conditioning"
# (result stem, label, video input, scorer, stage). Rows without results are left out.
ABLATIONS = (
    ("vjepa_baseline", r"Pooled states (B2)", r"$\bar z$", "--", "1"),
    ("ablation_mean_transition", r"\texttt{mean\_transition}", r"$\overline{\Delta z}$", r"$g$", "1"),
    ("ablation_final_minus_initial", r"\texttt{final\_minus\_initial}", r"$z_T - z_1$", r"$g$", "1"),
    ("ablation_transition_only", r"\texttt{transition\_only}", r"$\Delta z_{1..T-1}$", r"$g$", "1"),
    ("ablation_shuffled_states", r"\texttt{shuffled\_states}", r"$z_{\pi(1..T)}$", r"$g$", "1"),
    ("ablation_no_transition", r"\texttt{no\_transition}", r"$z_{1..T}$", r"$g$", "1"),
    ("transition_model", r"Single-stage", r"$[z_t,\Delta z_t]$", r"$g$", "1"),
    (STAGE1_E1, r"E1, stage 1", r"$[z_t,0]$", "--", "1"),
    (STAGE1_E2, r"E2, stage 1 (B3)", r"$[z_t,\Delta z_t]$", "--", "1"),
    (E1_RUN, r"E1", r"$[z_t,0]$", r"token", "2"),
    (E2_RUN, r"E2 (ours)", r"$[z_t,\Delta z_t]$", r"token", "2"),
)
# Sequence models evaluated with reversed states; rows without reversed results are left out.
REVERSED = (
    ("ablation_no_transition", r"\texttt{no\_transition}"),
    ("ablation_shuffled_states", r"\texttt{shuffled\_states}"),
    ("ablation_transition_only", r"\texttt{transition\_only}"),
    (STAGE1_E1, r"E1, stage 1"),
    (STAGE1_E2, r"E2, stage 1 (B3)"),
    (E1_RUN, "E1"),
    (E2_RUN, "E2 (ours)"),
)
# Macro prefix -> result stem, for the numbers quoted in the text.
MACRO_MODELS = {
    "Etwo": E2_RUN,
    "Eone": E1_RUN,
    "Bone": "clip_baseline",
    "Btwo": "vjepa_baseline",
    "Nt": "ablation_no_transition",
    "Tonly": "ablation_transition_only",
    "Shuf": "ablation_shuffled_states",
    "Single": "transition_model",
    "StageEtwo": STAGE1_E2,
    "StageEone": STAGE1_E1,
    "Mean": "ablation_mean_transition",
    "Fmi": "ablation_final_minus_initial",
}
REVERSED_MACRO_MODELS = ("Etwo", "Eone", "Nt", "Tonly", "Shuf", "StageEtwo", "StageEone")
# Error categories of the shared interface (Tickets T4.1) and the models compared.
ERROR_FILE = "error_analysis.json"
ERROR_CATEGORIES = (
    ("correct_objects_wrong_action", "Correct objects, wrong action"),
    ("wrong_temporal_direction", "Wrong temporal direction"),
<<<<<<< HEAD
=======
    ("wrong_action_ordering", "Wrong action ordering"),
>>>>>>> 7db566adf0a83061c85918e3a11ffd5cb7f10719
    ("similar_start_end_different_motion", "Similar start/end, different motion"),
    ("text_ambiguity", "Text ambiguity"),
    ("visual_ambiguity", "Visual ambiguity"),
    ("encoder_failure", "Encoder failure"),
)
ERROR_MODELS = (
    ("clip_baseline", "B1"),
    ("vjepa_baseline", "B2"),
    (E1_RUN, "E1"),
    (E2_RUN, "E2"),
)


# Loading

class Results:
    """Result folders; a file in a later folder replaces the same file in an earlier one."""

    def __init__(self, folders):
        self.folders = [Path(f) for f in ([folders] if isinstance(folders, (str, Path)) else folders)]
        if not self.folders:
            raise ValueError("at least one result folder is needed")

    def path(self, name: str) -> Path | None:
        found = [folder / name for folder in self.folders if (folder / name).exists()]
        return found[-1] if found else None

    def runs(self, stem: str, seeds=TRAIN_SEEDS, suffix: str = "") -> list[dict]:
        """Return <stem><suffix>.json, or else <stem>_seed<s><suffix>.json for every seed found."""
        if single := self.path(f"{stem}{suffix}.json"):
            return [load_json(single)]
        return [
            load_json(path)
            for seed in seeds
            if (path := self.path(f"{stem}_seed{seed}{suffix}.json"))
        ]

    def per_query(self, run: str, seed: int) -> list[dict]:
        name = f"per_query_{run}_seed{seed}.jsonl"
        path = self.path(name)
        if path is None:
            raise FileNotFoundError(f"{name} is in none of {self.folders}")
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metric(result: dict, setting: str, name: str, negative_type: str | None = None):
    """Return one metric as a percentage, or None if it was not computed."""
    block = result.get(setting)
    if block and negative_type:
        block = (block.get("by_negative_type") or {}).get(negative_type)
    value = (block or {}).get(name)
    return None if value is None else 100.0 * value


def mean_std(runs: list[dict], setting: str, name: str, negative_type: str | None = None):
    """Return (mean, sample std or None, number of runs) over the runs that have the metric."""
    values = [v for r in runs if (v := metric(r, setting, name, negative_type)) is not None]
    if not values:
        return None, None, 0
    std = float(np.std(values, ddof=1)) if len(values) > 1 else None
    return float(np.mean(values)), std, len(values)


# Paired comparison

def rank_metrics(rows: list[dict], setting: str) -> dict[str, float]:
    """Recall@1 and MRR (percent) over the queries that have a rank in this setting."""
    ranks = np.array([r[f"{setting}_rank"] for r in rows if r.get(f"{setting}_rank") is not None], dtype=float)
    return {"recall@1": 100.0 * float(np.mean(ranks == 1)), "mrr": 100.0 * float(np.mean(1.0 / ranks))}


def paired_comparison(
    results: Results,
    seeds=TRAIN_SEEDS,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """E2 - E1 per setting: per-seed Recall@1 differences with the shared paired bootstrap.

    Runs with the same seed are paired; each seed gets its own CI. MRR gets
    per-seed differences only, because the shared bootstrap covers Recall@1.
    """
    e2 = {s: results.per_query(E2_RUN, s) for s in seeds}
    e1 = {s: results.per_query(E1_RUN, s) for s in seeds}
    out = {"seeds": list(seeds), "bootstrap_samples": samples, "bootstrap_seed": seed,
           "method": "src.eval.evaluate.paired_bootstrap_recall_at_1, one bootstrap per seed",
           "settings": {}}
    for setting in SETTINGS:
        e1_metrics = {s: rank_metrics(e1[s], setting) for s in seeds}
        e2_metrics = {s: rank_metrics(e2[s], setting) for s in seeds}
        recall = {}
        for s in seeds:
            ci = paired_bootstrap_recall_at_1(
                e1[s], e2[s], rank_key=f"{setting}_rank", n_bootstrap=samples, seed=seed
            )
            recall[str(s)] = {
                "difference": ci["difference_pp"],
                "ci95": [ci["ci_low_pp"], ci["ci_high_pp"]],
                "num_queries": ci["num_queries"],
            }
        lows = [r["ci95"][0] for r in recall.values()]
        highs = [r["ci95"][1] for r in recall.values()]
        block = {"num_queries": recall[str(seeds[0])]["num_queries"]}
        for name in ("recall@1", "mrr"):
            per_seed = {str(s): e2_metrics[s][name] - e1_metrics[s][name] for s in seeds}
            block[name] = {
                "e1": float(np.mean([e1_metrics[s][name] for s in seeds])),
                "e2": float(np.mean([e2_metrics[s][name] for s in seeds])),
                "e1_per_seed": [e1_metrics[s][name] for s in seeds],
                "e2_per_seed": [e2_metrics[s][name] for s in seeds],
                "difference": float(np.mean(list(per_seed.values()))),
                "per_seed": per_seed,
            }
        block["recall@1"]["bootstrap"] = recall
        block["recall@1"]["criterion_met"] = bool(min(lows) > 0 or max(highs) < 0)
        out["settings"][setting] = block
    return out


def check_against_saved(results: Results, comparison: dict, seeds=TRAIN_SEEDS) -> None:
    """Fail if a metric recomputed from per-query ranks differs from the saved JSON."""
    for run, key in ((E2_RUN, "e2_per_seed"), (E1_RUN, "e1_per_seed")):
        for i, s in enumerate(seeds):
            saved = load_json(results.path(f"{run}_seed{s}.json"))
            for setting, block in comparison["settings"].items():
                for name in ("recall@1", "mrr"):
                    recomputed, expected = block[name][key][i], metric(saved, setting, name)
                    if abs(recomputed - expected) > 100.0 * CHECK_TOLERANCE:
                        raise ValueError(
                            f"{run} seed {s} {setting} {name}: per-query {recomputed:.4f} "
                            f"!= saved {expected:.4f}"
                        )
                bootstrap = block["recall@1"]["bootstrap"][str(s)]["difference"]
                if abs(bootstrap - block["recall@1"]["per_seed"][str(s)]) > 100.0 * CHECK_TOLERANCE:
                    raise ValueError(f"seed {s} {setting}: bootstrap and per-query differences disagree")


# LaTeX

def fmt(mean, std=None, digits: int = 1, bold: bool = False) -> str:
    if mean is None:
        return "--"
    text = f"{mean:.{digits}f}"
    if bold:
        text = rf"\textbf{{{text}}}"
    return rf"\ms{{{text}}}{{{std:.{digits}f}}}" if std is not None else text


def signed(value: float, digits: int = 2) -> str:
    return f"{value:+.{digits}f}".replace("-", "$-$")


def table_rows(rows: list[tuple[str, list[dict]]], columns: list[tuple]) -> list[str]:
    """Format rows of mean +- std; the best mean of each column is bold."""
    cells = [[mean_std(runs, *column) for column in columns] for _, runs in rows]
    best = [max((c[j][0] for c in cells if c[j][0] is not None), default=None) for j in range(len(columns))]
    lines = []
    for (label, _), row in zip(rows, cells):
        values = [
            fmt(m, s, bold=m is not None and best[j] is not None and abs(m - best[j]) < 1e-9)
            for j, (m, s, _) in enumerate(row)
        ]
        lines.append(" & ".join([label, *values]) + r" \\")
    return lines


def write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("% Generated by src/models/transition_model/pipelines/report.py; do not edit.\n"
                    + "\n".join(lines) + "\n", encoding="utf-8")


def main_table(results: Results) -> list[str]:
    rows = [(label, results.runs(stem)) for stem, label in BASELINES]
    rows += [
        (r"\texttt{no\_transition}", results.runs("ablation_no_transition")),
        (r"E1: states only", results.runs(E1_RUN)),
        (r"E2: states + $\Delta z$ (ours)", results.runs(E2_RUN)),
    ]
    columns = [
        ("full", "recall@1"), ("full", "recall@5"), ("full", "recall@10"), ("full", "mrr"),
        ("hard_negative", "recall@1"), ("hard_negative", "mrr"),
        ("action_contrast", "recall@1"), ("action_contrast", "mrr"),
    ]
    body = table_rows(rows, columns)
    return [*body[:-2], r"\midrule", *body[-2:]]


def paired_table(comparison: dict) -> list[str]:
    seeds = comparison["seeds"]
    lines = []
    for setting, block in comparison["settings"].items():
        c = block["recall@1"]
        cells = []
        for s in seeds:
            b = c["bootstrap"][str(s)]
            cells += [signed(b["difference"]), f"[{signed(b['ci95'][0])}, {signed(b['ci95'][1])}]"]
        mark = r"\checkmark" if c["criterion_met"] else "--"
        lines.append(" & ".join([SETTING_NAMES[setting], fmt(c["e1"]), fmt(c["e2"]), *cells, mark]) + r" \\")
    return lines


def bytype_table(results: Results) -> list[str]:
    rows = [
        ("B1: CLIP", results.runs("clip_baseline")),
        ("B2: V-JEPA~2", results.runs("vjepa_baseline")),
        (r"\texttt{no\_transition}", results.runs("ablation_no_transition")),
        (r"E1", results.runs(E1_RUN)),
        (r"E2 (ours)", results.runs(E2_RUN)),
    ]
    columns = [("hard_negative", "pairwise_accuracy", t) for t in NEGATIVE_TYPES]
    columns.append(("action_contrast", "pairwise_accuracy", "opposite_action"))
    body = table_rows(rows, columns)
    return [*body[:-2], r"\midrule", *body[-2:]]


def ablation_table(results: Results) -> list[str]:
    columns = [("full", "mrr"), ("hard_negative", "mrr"), ("action_contrast", "recall@1")]
    lines = []
    for stem, label, video_input, scorer, stage in ABLATIONS:
        runs = results.runs(stem)
        if not runs:
            continue
        if stem == E1_RUN:
            lines.append(r"\midrule")
        cells = [fmt(*mean_std(runs, *column)[:2]) for column in columns]
        lines.append(" & ".join([label, video_input, scorer, stage, str(len(runs)), *cells]) + r" \\")
    return lines


def reverse_table(results: Results) -> list[str]:
    columns = [("full", "mrr"), ("hard_negative", "mrr"), ("action_contrast", "recall@1")]
    lines = []
    for stem, label in REVERSED:
        forward, reverse = results.runs(stem), results.runs(stem, suffix="_reversed")
        if not forward or not reverse:
            continue
        if stem == E1_RUN and lines:
            lines.append(r"\midrule")
        cells = []
        for column in columns:
            f, r = mean_std(forward, *column)[0], mean_std(reverse, *column)[0]
            cells += [fmt(f), fmt(r), "--" if f is None or r is None else signed(r - f, 1)]
        lines.append(" & ".join([label, str(len(reverse)), *cells]) + r" \\")
    return lines


def latex_escape(text: str) -> str:
    special = {"\\": r"\textbackslash{}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(special.get(c, "\\" + c if c in "&%$#_{}" else c) for c in text)


def error_section(errors: dict) -> list[list[str]] | None:
    """Return (table rows, subsection lines) from the T4.1 output, or None if no model is in it.

    Shares are of each model's failed full-retrieval queries; examples are the
    first E2 example of each observed category (query, target, top retrieved).
    """
    summary = errors.get("summary", {})
    models = [(name, label) for name, label in ERROR_MODELS if name in summary]
    if not models:
        return None
    rows = []
    for key, label in ERROR_CATEGORIES:
        cells = []
        for name, _ in models:
            block = summary[name]
            share = 100.0 * block["counts"].get(key, 0) / max(block["num_errors"], 1)
            cells.append(f"{share:.1f}")
        rows.append(" & ".join([label, *cells]) + r" \\")
    failures = " & ".join(
        ["Failed queries", *(f"{summary[n]['num_errors']:,}".replace(",", "{,}") for n, _ in models)]
    )
    examples = []
    e2_examples = summary.get(E2_RUN, {}).get("examples", {})
    for key, label in ERROR_CATEGORIES:
        for example in e2_examples.get(key, [])[:1]:
            examples.append(
                rf"\item \emph{{{label}}}: ``{latex_escape(str(example.get('text_query', '')))}'' "
                rf"(target {latex_escape(str(example.get('target_video_id', '')))}, rank "
                rf"{example.get('target_rank', '--')}; top retrieved "
                rf"{latex_escape(str(example.get('retrieved_video_id', '')))}"
                + (rf", ``{latex_escape(str(example['retrieved_label']))}''" if example.get("retrieved_label") else "")
                + ")."
            )
    seeds = sorted({m.group(1) for f in errors.get("ranking_files", []) if (m := re.search(r"_seed(\d+)", f))})
    seed_note = f" Trained models: seed {', '.join(seeds)}." if seeds else ""
    lines = [
        r"\subsection{Error Analysis}",
        r"\label{sec:errors}",
        r"For each query whose target is not ranked first in full retrieval, the",
        r"top-ranked wrong video is compared with the target using SSv2 templates,",
<<<<<<< HEAD
        r"labels, and object names, and the failure is assigned to one of six",
=======
        r"labels, and object names, and the failure is assigned to one of seven",
>>>>>>> 7db566adf0a83061c85918e3a11ffd5cb7f10719
        r"categories. The rules sort failures for inspection; they are heuristics, not",
        r"ground truth. Table~\ref{tab:errors} gives the share of each category.",
        r"\begin{table}[!t]",
        r"\caption{Categories of failed full-retrieval queries (share of each model's",
        r"failures, percent)." + seed_note + "}",
        r"\label{tab:errors}",
        r"\centering",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\begin{{tabular}}{{l{'c' * len(models)}}}",
        r"\toprule",
        " & ".join(["Category", *(label for _, label in models)]) + r" \\",
        r"\midrule",
        r"\tabinput{generated/tab_errors.tex}",
        r"\midrule",
        failures + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    if examples:
        lines += [r"Representative E2 failures, one per observed category:", r"\begin{itemize}",
                  *examples, r"\end{itemize}"]
    return [rows, lines]


def number_macros(results: Results, comparison: dict) -> list[str]:
    """Macros for every number the paper text quotes (percentages, no % sign)."""
    short = {"full": "Full", "hard_negative": "Hn", "action_contrast": "Ac"}
    metric_short = {"recall@1": "Rone", "recall@5": "Rfive", "recall@10": "Rten", "mrr": "Mrr"}
    models = {prefix: results.runs(stem) for prefix, stem in MACRO_MODELS.items()}
    models.update({
        prefix + "Rev": results.runs(MACRO_MODELS[prefix], suffix="_reversed")
        for prefix in REVERSED_MACRO_MODELS
    })
    lines = []

    def define(name: str, value) -> None:
        lines.append(rf"\newcommand{{\{name}}}{{{value}}}")

    for model, runs in models.items():
        if runs and not model.endswith("Rev"):
            define(model + "Runs", len(runs))
        for setting in SETTINGS:
            for name in metric_short:
                mean, _, n = mean_std(runs, setting, name)
                if n:
                    define(model + short[setting] + metric_short[name], f"{mean:.1f}")
        for negative_type, tag in zip(NEGATIVE_TYPES, ("Opp", "Same", "Other")):
            mean, _, n = mean_std(runs, "hard_negative", "pairwise_accuracy", negative_type)
            if n:
                define(model + "Pair" + tag, f"{mean:.1f}")
        mean, _, n = mean_std(runs, "action_contrast", "pairwise_accuracy", "opposite_action")
        if n:
            define(model + "AcPair", f"{mean:.1f}")
    for setting, block in comparison["settings"].items():
        for name in ("recall@1", "mrr"):
            c = block[name]
            prefix = "Delta" + short[setting] + metric_short[name]
            per_seed = list(c["per_seed"].values())
            define(prefix, signed(c["difference"]))
            define(prefix + "SeedMin", signed(min(per_seed)))
            define(prefix + "SeedMax", signed(max(per_seed)))
        bootstrap = block["recall@1"]["bootstrap"]
        prefix = "Delta" + short[setting] + "Rone"
        for s, b in bootstrap.items():
            word = SEED_WORDS.get(int(s), "Seed" + "".join(chr(ord("a") + int(d)) for d in s))
            define(prefix + word, signed(b["difference"]))
            define(prefix + "Low" + word, signed(b["ci95"][0]))
            define(prefix + "High" + word, signed(b["ci95"][1]))
        define(prefix + "LowMin", signed(min(b["ci95"][0] for b in bootstrap.values())))
        define(prefix + "HighMax", signed(max(b["ci95"][1] for b in bootstrap.values())))
    for name, key in (("EtwoEpochs", "Etwo"), ("EoneEpochs", "Eone")):
        define(name, ", ".join(str(r.get("best_epoch")) for r in models[key]))
    define("BootstrapSamples", f"{comparison['bootstrap_samples']:,}".replace(",", "{,}"))
    return lines


# Main function

def build_report(result_dirs, output_dir: Path, samples: int = BOOTSTRAP_SAMPLES) -> dict:
    results = Results(result_dirs)
    comparison = paired_comparison(results, samples=samples)
    check_against_saved(results, comparison)
    (results.folders[0] / "e1_e2_paired_bootstrap.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    write(output_dir / "numbers.tex", number_macros(results, comparison))
    write(output_dir / "tab_main.tex", main_table(results))
    write(output_dir / "tab_paired.tex", paired_table(comparison))
    write(output_dir / "tab_bytype.tex", bytype_table(results))
    write(output_dir / "tab_ablation.tex", ablation_table(results))
    write(output_dir / "tab_reverse.tex", reverse_table(results))
    # The error-analysis subsection exists only when the T4.1 output does.
    error_path = results.path(ERROR_FILE)
    section = error_section(load_json(error_path)) if error_path else None
    if section is None:
        for name in ("tab_errors.tex", "sec_errors.tex"):
            (output_dir / name).unlink(missing_ok=True)
    else:
        rows, lines = section
        write(output_dir / "tab_errors.tex", rows)
        write(output_dir / "sec_errors.tex", lines)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result-dir", type=Path, nargs="+", default=[RESULT_ROOT],
                        help="result folders; a file in a later folder replaces the same file in an earlier one")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "paper" / "generated")
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    args = parser.parse_args()
    comparison = build_report(args.result_dir, args.output_dir, args.bootstrap_samples)
    for setting, block in comparison["settings"].items():
        c = block["recall@1"]
        seeds = "  ".join(
            f"{s}: {b['difference']:+.2f} [{b['ci95'][0]:+.2f}, {b['ci95'][1]:+.2f}]"
            for s, b in c["bootstrap"].items()
        )
        print(f"{setting:16s} R@1 E1 {c['e1']:6.2f}  E2 {c['e2']:6.2f}  {seeds}  "
              f"criterion {'met' if c['criterion_met'] else 'not met'}")
    print(f"wrote {args.output_dir}/ and {args.result_dir[0] / 'e1_e2_paired_bootstrap.json'}")


if __name__ == "__main__":
    main()
