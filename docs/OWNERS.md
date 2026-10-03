# Ownership Boundaries

To reduce merge conflicts, each teammate should mostly edit their own folder.

## Priyanka

- `src/data/`
- `src/eval/`
- dataset splits, hard negatives, metrics, and dataset statistics

## Quynh

- `config.py`
- `src/models/transition_model/`
- V-JEPA feature extraction, text encoding, transitions, retrieval model, and main pipelines

## Hao

- `src/models/baselines/`
- CLIP baseline, V-JEPA baseline, ablation experiment runners, and qualitative error analysis

## Shared Interfaces

- Quynh exposes transition-model caches, checkpoints, and scoring helpers needed for Hao's ablations.
- Priyanka owns evaluator behavior and metric definitions used by every model.
- Hao may call Quynh's transition helpers and Priyanka's evaluator, but should avoid copying that logic into baseline files.

## Shared Rule

Avoid changing another teammate's folder unless you ask first or the change is purely an import/path update needed by your own code.
