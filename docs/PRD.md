# PRD — Accumulated Temporal Transitions for Text-Video Retrieval

## 1. Project Overview

**Project:** Accumulated Temporal Transitions for Text-Video Retrieval  
**Course:** ECE 6356 — Fall 2026  
**Dataset:** Something-Something V2  
**Primary task:** Text-to-video retrieval

### Research Question

Can modeling temporal changes accumulated across a video improve text-to-video retrieval, especially for visually similar videos with different actions or outcomes?

### Motivation

Many videos contain the same objects and similar visual appearance but differ in how the scene changes over time.

Examples:

- **Pick up:** cup on table → cup in hand
- **Put down:** cup in hand → cup on table
- **Move closer:** cup far away → cup nearby

A retrieval system should distinguish the action and temporal evolution, not only recognize the objects present in the scene.

---

## 2. Project Goal

Build and evaluate a text-video retrieval system that explicitly models latent temporal transitions in video representations.

The project will test whether a transition-aware representation improves retrieval compared with appearance-based and non-transition baselines.

---

## 3. Proposed Method

### 3.1 Video Encoding

Use **V-JEPA 2** to extract latent representations from sampled video frames.

Given sampled temporal states:

\[
z_1, z_2, ..., z_T
\]

### 3.2 Temporal State Sampling

Sample **8–16 ordered frames/states** from each video.

The sampling order must preserve temporal sequence.

### 3.3 Latent Transition Computation

Compute consecutive latent changes:

\[
\Delta z_t = z_{t+1} - z_t
\]

for:

\[
t = 1, ..., T-1
\]

This produces the transition sequence:

\[
[\Delta z_1, \Delta z_2, ..., \Delta z_{T-1}]
\]

### 3.4 Transition Sequence Encoding

Encode the transition sequence using a temporal module such as temporal attention.

The output should summarize how the video evolves over time.

### 3.5 Query-Conditioned Matching

Encode the text query and combine it with the accumulated temporal representation.

Use a trained reranker to score candidate videos for each query.

---

## 4. Dataset

### Something-Something V2

Current project assumptions from the preliminary proposal:

- 220,847 labeled video clips
- 174 action categories
- 318,572 object-filled annotations
- Short VP9 clips
- Approximately 12 FPS

### Retrieval Task

Use action/object descriptions as text queries.

**Input:**
- Video
- Action/object text description

**Target:**
- The matching video

### Data Split

- Train: 168,913 clips
- Official labeled validation: 24,777 clips
- Split the labeled validation set into:
  - development validation set
  - held-out test set

Do not use the official hidden-label test set for final evaluation.

---

## 5. Hard Negatives

Hard negatives are a central part of the evaluation.

Construct candidate videos that are visually similar or share the same objects but represent different actions, directions, or outcomes.

Examples:

- pick up vs put down
- move closer vs move farther
- open vs close
- push left vs push right

The hard-negative benchmark should test whether temporal transitions provide information beyond static appearance.

---

## 6. Baselines

### B1 — CLIP + Frame Averaging

- Encode sampled frames with CLIP.
- Average frame representations.
- Match averaged video representation with text embedding.

### B2 — V-JEPA 2 Without Transition Cue

- Use V-JEPA 2 video representations.
- Do not explicitly compute or encode latent transitions.

### B3 — Reranker Without Query Conditioning

- Use temporal representation.
- Remove query-conditioned interaction from the reranker.

### B4 — Transition Ablation

Remove or replace the latent transition sequence to measure its contribution.

Possible variants:

- No transition sequence
- Mean-pooled transition vector
- Final minus initial state only

---

## 7. Evaluation

### Primary Metrics

- Recall@1
- Recall@5
- Recall@10
- Mean Reciprocal Rank (MRR)
- mean Average Precision (mAP)

### Evaluation Settings

1. Full retrieval benchmark
2. Hard-negative retrieval benchmark
3. Ablation experiments

### Main Success Criterion

The transition-aware model should improve retrieval metrics over relevant baselines, with particular attention to hard-negative examples where videos share similar objects or appearance.

---

## 8. Error Analysis

Analyze failure cases by category.

Recommended categories:

- Correct objects, wrong action
- Correct action family, wrong temporal direction
- Wrong ordering of states
- Similar start/end states but different intermediate motion
- Text ambiguity
- Visually ambiguous clip
- Encoder representation failure

Qualitative examples should show:

- Query
- Ground-truth video
- Top retrieved video(s)
- Baseline result
- Transition-model result

---

## 9. System Interfaces

To allow independent development, modules should communicate through simple shared interfaces.

```text
dataset.py
    -> video_id
    -> frames
    -> text_query
    -> target_video_id

encoder.py
    frames -> z[1:T]

transition.py
    z[1:T] -> delta_z[1:T-1]

retrieval.py
    video_representation + text_embedding -> score

evaluate.py
    scores + targets -> R@1, R@5, R@10, MRR, mAP
```

Intermediate embeddings should be cacheable so baselines and ablations do not repeatedly run expensive encoders.

---

## 10. Team Ownership

### Quynh T Nguyen

Primary ownership:

- Model design
- Main implementation
- V-JEPA 2 feature pipeline
- Latent transition module
- Query-conditioned reranker
- Main experiments
- Final integration

### Priyanka Subramanian

Primary ownership:

- Data preparation
- Benchmark construction
- Train/validation/test split
- Hard-negative construction
- Evaluation implementation
- Dataset statistics
- Metric validation

### Hao T Nguyen

Primary ownership:

- Baseline implementation
- Ablation studies
- Error analysis
- Qualitative visualization
- Baseline/model comparison

---

## 11. Milestones

### Milestone 1 — Data and Baseline Infrastructure

Deliverables:

- Dataset loader
- Reproducible data split
- Frame sampling
- CLIP baseline
- V-JEPA feature extraction
- Evaluation script

### Milestone 2 — Transition Model

Deliverables:

- Transition computation
- Transition sequence encoder
- Query-conditioned reranker
- End-to-end training pipeline

### Milestone 3 — Experiments

Deliverables:

- Full retrieval results
- Hard-negative results
- Baseline comparison
- Ablations

### Milestone 4 — Analysis and Final Report

Deliverables:

- Error analysis
- Qualitative examples
- Final tables
- Discussion
- Limitations
- Presentation-ready results

---

## 12. Risks

### Novelty Risk

The difference operation itself is simple.

The project must demonstrate that the transition sequence provides useful information beyond ordinary temporal video representations.

### Dataset Risk

Something-Something V2 is short and controlled, with templated language.

Results should not be presented as evidence for open-domain video search.

### Compute Risk

Repeated V-JEPA 2 encoding may be expensive.

Mitigation:

- cache video embeddings
- reuse sampled frame indices
- separate feature extraction from downstream experiments

### Evaluation Risk

Overall retrieval metrics may hide the intended benefit.

Hard-negative evaluation is therefore required.

---

## 13. Definition of Done

The project is complete when:

- The dataset pipeline is reproducible.
- At least two strong baselines run end-to-end.
- The proposed transition-aware model runs end-to-end.
- Recall@1/5/10, MRR, and mAP are reported.
- Hard-negative performance is reported separately.
- At least one transition ablation is completed.
- Error analysis includes representative qualitative examples.
- Final results can be reproduced from documented commands/configuration.
