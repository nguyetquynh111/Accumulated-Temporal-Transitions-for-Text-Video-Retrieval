# Accumulated Temporal Transitions for Text-Video Retrieval

**Research question.** Can accumulated latent transitions between consecutive
video states improve text-video retrieval compared with static video
representations?

**Method.** Each video becomes a sequence of V-JEPA 2 states `z[1..T]`. We add
the transition between consecutive states, `Δz[t] = z[t+1] - z[t]`, encode the
tokens `[z[t], Δz[t]]` with a temporal self-attention encoder, and score them
against the text query with a token-level cross-attention matcher. Training has
two stages: the encoder first, then the matcher on the frozen encoder.

**Experiment.** The question is answered by a matched comparison on
Something-Something V2 (SSv2):

- **E2 (proposed):** tokens `[z[t], Δz[t]]`.
- **E1 (control):** the same model, data, training, and seeds, with tokens
  `[z[t], 0]`. Only `Δz` differs.

Both are evaluated on full retrieval (12,389 test videos), a hard-negative set,
and an action-contrast set (opposite-action distractors).

**Answer (seeds 42, 43, 44).** Yes for action-sensitive retrieval:
action-contrast Recall@1 rises from 72.1% (E1) to 75.9% (E2), and the
paired-bootstrap 95% CI is above zero for every seed. Hard-negative Recall@1
rises from 77.0% to 78.1%. On full retrieval there is no reliable gain
(MRR 19.5 → 19.9).

## 1. Set up the environment

```bash
git clone https://github.com/nguyetquynh111/Accumulated-Temporal-Transitions-for-Text-Video-Retrieval.git
cd Accumulated-Temporal-Transitions-for-Text-Video-Retrieval

conda create -n attvr python=3.11 -y
conda activate attvr
pip install -r requirements.txt

python -m pytest          # should pass; needs no data or GPU
```

Model weights (V-JEPA 2, CLIP, MiniLM) download automatically from Hugging Face
to `outputs/model_cache/` the first time they are used.

## 2. Download the dataset

The train/val/test splits and the hard-negative sets are already in Git
(`data/annotations/`). You only need the SSv2 videos.

1. Create an account and accept the license on the
   [SSv2 page](https://www.qualcomm.com/developer/software/something-something-v-2-dataset).
2. Download the video parts `20bn-something-something-v2-00` to `-19`.
3. Extract the videos and link them into the project:

   ```bash
   cat 20bn-something-something-v2-?? | tar -xvzf -     # -> 20bn-something-something-v2/<id>.webm
   ln -s /path/to/20bn-something-something-v2 data/videos
   ```

   `data/videos/` should then contain 220,847 `.webm` files.

## 3. Run the main commands

Run from the repository root with `conda activate attvr`. Steps 3.1–3.4 need a
GPU. Add `--limit 200` to most commands for a quick test on a small sample.

### 3.1 Extract V-JEPA 2 features

```bash
python -m src.models.transition_model.pipelines.extract_features \
  --annotations data/annotations/train.jsonl data/annotations/val.jsonl data/annotations/test.jsonl
```

Writes one file per video to `outputs/features/vjepa/`. If it stops, run it
again; finished videos are skipped.

### 3.2 Baselines and ablations

```bash
python -m src.models.baselines.clip_baseline                                          # B1: CLIP
python -m src.models.baselines.vjepa_baseline --variant pooled all_ablations --seeds 42 43 44
python -m src.models.baselines.vjepa_baseline --variant all_ablations --reverse-time --seeds 42 43 44
```

The second command trains B2 (pooled V-JEPA) and the six ablations. One of them,
`full_transition_without_query_conditioning`, is stage 1 of E2.

### 3.3 Main experiment: E1 vs E2

```bash
# E1 stage 1: states-only encoder
for seed in 42 43 44; do
  python -m src.models.transition_model.pipelines.train --seed "$seed" \
    --run-name ablation_state_only_without_query_conditioning \
    --video-input state_only --no-query-conditioning
done
python -m src.models.transition_model.pipelines.evaluate --run-name ablation_state_only_without_query_conditioning

# Stage 2: token-level matcher for E2 and E1, then reverse-time evaluation
for encoder in full_transition state_only; do
  python -m src.models.transition_model.pipelines.train_token_reranker --encoder "$encoder" --seeds 42 43 44
  python -m src.models.transition_model.pipelines.train_token_reranker --encoder "$encoder" --reverse-time --seeds 42 43 44
done
```

### 3.4 Single-stage model (for the ablation table)

```bash
for seed in 42 43 44; do
  python -m src.models.transition_model.pipelines.train --seed "$seed"
done
python -m src.models.transition_model.pipelines.evaluate
```

### 3.5 Tables, statistics, and paper (CPU)

```bash
# Error analysis and example figures
python -m src.models.baselines.error_analysis --figure-dir outputs/results/qualitative

# Comparison table (mean ± std over seeds)
R=outputs/results
python -m src.eval.evaluate \
  $(ls $R/{clip_baseline,vjepa_baseline,ablation_,transition_model,two_stage_}*.json | grep -v _reversed) \
  --output $R/comparison_table.csv

# E1 vs E2 paired bootstrap + LaTeX tables, then the PDF
python -m src.models.transition_model.pipelines.report
cd paper && latexmk -pdf main.tex
```

The comparison table and `report.py` read the results already in Git, so they
run without steps 3.1–3.4.

### All steps at once

```bash
bash scripts/reproduce_all.sh 2     # steps 3.1–3.4
```

## 4. Run on Slurm

Submit from the repository root, in this order, waiting for each job to finish:

```bash
bash scripts/submit_single_stage.sh            # 3.1 features + 3.4 single-stage model
bash scripts/submit_baselines.sh               # 3.2 baselines and ablations
sbatch scripts/train_token_matcher.sbatch --encoder full_transition --seeds 42 43 44   # E2 stage 2
bash scripts/submit_main_experiment.sh         # 3.3 E1, E1/E2 evaluation, reverse time
sbatch scripts/build_tables.sbatch             # 3.5 comparison table and error analysis
```

Jobs activate the `attvr` env through `scripts/env.sh` (change it with
`ATTVR_CONDA_ENV`) and write their logs to `logs/`. Use
`bash scripts/submit_single_stage.sh --skip-extract` if features already exist.

## 5. Where things are

| Path | Content | In Git |
|---|---|---|
| `config.py` | All paths and training settings | yes |
| `data/annotations/` | Splits, hard-negative and action-contrast sets | yes |
| `data/videos/` | SSv2 videos | no |
| `outputs/features/`, `outputs/checkpoints/` | Features and trained models | no |
| `outputs/results/*.json` | Metrics for every model and seed | yes |
| `outputs/results/per_query_*.jsonl` | Per-query ranks (E1/E2 in Git, others not) | partly |
| `outputs/results/rankings_*.jsonl` | Top retrieved videos per query | no |
| `outputs/results/qualitative/` | Error-analysis figures | yes |
| `outputs/reports/` | Result write-ups | yes |
| `paper/` | LaTeX paper; `paper/generated/` is written by `report.py` | yes |
| `logs/` | Training logs (Slurm `.out` files are not in Git) | partly |

Each metrics JSON reports Recall@1/5/10, MRR, and mAP in three settings:
`full` (all 12,389 test videos), `hard_negative` (target vs. up to 9 hard
negatives), and `action_contrast` (target vs. opposite-action videos).

## 6. Configuration

Settings live in `config.py`: 16 frames per video, batch size 128, learning
rate 1e-4, at most 50 epochs with early stopping on validation MRR (patience 5),
and seeds 42, 43, 44. To move data or outputs elsewhere, set environment
variables instead of editing code, for example:

```bash
export ATTVR_FEATURE_ROOT=/scratch/$USER/attvr/features
export ATTVR_CHECKPOINT_ROOT=/scratch/$USER/attvr/checkpoints
```

Other variables: `ATTVR_DATA_ROOT`, `ATTVR_ANNOTATION_ROOT`, `ATTVR_VIDEO_ROOT`,
`ATTVR_OUTPUT_ROOT`, `ATTVR_RESULT_ROOT`, `ATTVR_MODEL_CACHE_ROOT`,
`ATTVR_LOG_ROOT`, `ATTVR_RANDOM_SEED`.

Every command accepts `--help` for all its options.

## 7. Tests

```bash
python -m pytest
```

## Team documents

- [docs/PRD.md](docs/PRD.md): goals and success criterion
- [docs/Tickets.md](docs/Tickets.md): tasks and data formats
- [docs/OWNERS.md](docs/OWNERS.md): who owns which folder
- [outputs/reports/t3_2_statistical_results.md](outputs/reports/t3_2_statistical_results.md): E1 vs E2 statistics
