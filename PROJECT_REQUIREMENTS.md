# Project requirements — models, data, and outputs

> **Python dependencies (pip):** use [`requirements.txt`](requirements.txt), not this file.

This document defines what this repository must provide: which **models** are in scope, which **data** they consume, which **scripts** produce them, and which **output directories** are required for the main report vs the QA appendix. It is derived from the current scripts and committed output folders, not from a generic deployment template.

For run commands and CLI flags, see [`README.md`](README.md) and [`README_SEMANTIC_SIGNAL_RELIABILITY.md`](README_SEMANTIC_SIGNAL_RELIABILITY.md).

---

## 1. Scope

| In scope | Out of scope |
|----------|----------------|
| Answer-level binary hallucination detection on HaluEval (knowledge + question + answer text) | Live LLM API integration at generation time |
| Local training and evaluation (Baseline, Proposed) | Multi-user hosted service or account management |
| External **reference** baselines (SummaC, HaluEval GPT-3.5 judge) on the **same splits** | Retraining or fine-tuning SummaC / judge models |
| Reproducible `predictions.json` + metrics + comparison figures | Production backup/DR infrastructure |

**Default data root:** `Data/`  
**Default split:** stratified 80% train / 10% val / 10% test, `--seed 42`

---

## 2. Data requirements (`Data/`, `halueval_baseline/`)

### 2.1 HaluEval training/evaluation data (`Data/`)

Required for `train_failure_aware.py`, `train_failure_aware_semantic_signal_reliability.py`, and `run_summac_baseline.py` (pass `--data_dir Data`).

| File | Sub-task | Used when |
|------|----------|-----------|
| `qa_data.json` | QA | `dataset_type=all` or `qa` |
| `dialogue_data.json` | Dialogue | `dataset_type=all` or `dialogue` |
| `summarization_data.json` | Summarization | `dataset_type=all` or `summarization` |
| `general_data.json` | General | `dataset_type=all` or `general` |

Each record supplies a prompt string (knowledge / question / answer fields per task) and a binary label: **0 = correct**, **1 = hallucinated**.

### 2.2 HaluEval official judge results (`halueval_baseline/`)

Required only for the **QA appendix** comparison (LLM-as-judge reference).

| File | Purpose |
|------|---------|
| `qa_gpt-3.5-turbo_result.json` | Official HaluEval QA hallucination-recognition output |
| `build_halueval_baseline.py` | Converts the above into `predictions.json` aligned with the same split as training |

Build command (must use the same `--data_dir` and `--seed` as QA training):

```bash
python halueval_baseline/build_halueval_baseline.py \
  --dataset_type qa \
  --result_file halueval_baseline/qa_gpt-3.5-turbo_result.json \
  --data_dir Data \
  --output_dir outputs_halueval_baseline_qa
```

---

## 3. Models and scripts

### 3.1 Learned models (trained in this repo)

| ID | Script | Description | Default output directory |
|----|--------|-------------|---------------------------|
| **Baseline** | `train_failure_aware.py` | DistilBERT-scale encoder + uncertainty features + calibration. **No** `SignalReliabilityAnalyzer`. | `outputs_failure_aware/` (all tasks) |
| **Proposed** | `train_failure_aware_semantic_signal_reliability.py` | Same core stack + **Stage 2.5** `SignalReliabilityAnalyzer` (trust tier, conflict reasoning, trust-aware confidence). Supports `--mode inference` for one prompt. | `outputs_failure_aware_semantic_signal_reliability/` (all tasks) |

**Delivery training command (all tasks, report metrics):**

```bash
python train_failure_aware.py --stage 3 --data_dir Data --output_dir outputs_failure_aware --seed 42

python train_failure_aware_semantic_signal_reliability.py --stage 3 \
  --data_dir Data \
  --output_dir outputs_failure_aware_semantic_signal_reliability \
  --seed 42 \
  --calibration_method temperature
```

**QA-only variants (appendix):**

| Model | Output directory |
|-------|------------------|
| Baseline-QA | `outputs_failure_aware_qa/` |
| Proposed-QA | `outputs_failure_aware_semantic_signal_reliability_qa/` |

```bash
python train_failure_aware.py --stage 3 --dataset_type qa --data_dir Data \
  --output_dir outputs_failure_aware_qa --seed 42

python train_failure_aware_semantic_signal_reliability.py --stage 3 --dataset_type qa \
  --data_dir Data --output_dir outputs_failure_aware_semantic_signal_reliability_qa --seed 42
```

### 3.2 External reference baselines (not trained here)

| ID | Script | Description | Default output directory |
|----|--------|-------------|---------------------------|
| **SummaC** | `run_summac_baseline.py` | Published NLI consistency scores; mapped to P(hallucination) via validation min–max (see script). Uses same `load_all_data` + `split_stratified` as Proposed. | `outputs_summac_baseline/` (all tasks) |
| **SummaC-QA** | `run_summac_baseline.py --dataset_type qa` | Same, QA sub-task only | `outputs_summac_baseline_qa/` |
| **HaluEval judge** | `halueval_baseline/build_halueval_baseline.py` | Official GPT-3.5-turbo QA judge probabilities | `outputs_halueval_baseline_qa/` |

```bash
python run_summac_baseline.py --data_dir Data --dataset_type all \
  --output_dir outputs_summac_baseline --seed 42

python run_summac_baseline.py --data_dir Data --dataset_type qa \
  --output_dir outputs_summac_baseline_qa --seed 42
```

### 3.3 Comparison scripts (no training)

| Script | Inputs (default dirs) | Output directory |
|--------|----------------------|------------------|
| `compare_baseline.py` | `outputs_failure_aware`, `outputs_failure_aware_semantic_signal_reliability`, optional `outputs_summac_baseline` | `outputs_comparison/` |
| `qa_compare_baseline.py` | `outputs_failure_aware_qa`, `outputs_failure_aware_semantic_signal_reliability_qa`, `outputs_halueval_baseline_qa`, optional `outputs_summac_baseline_qa` | `outputs_comparison_qa/` |

```bash
python compare_baseline.py
python qa_compare_baseline.py
```

**Alignment requirement:** every `predictions.json` used in a comparison must have the **same** `labels` length and order → same `--data_dir`, `--seed`, `--val_ratio`, `--test_ratio`, and `dataset_type`.

---

## 4. Minimum artefacts per output directory

### 4.1 Any trained or baseline scoring run

These files are **required** before running `compare_baseline.py` or `qa_compare_baseline.py`:

| File | Requirement |
|------|-------------|
| `predictions.json` | Parallel arrays `probs` (P(hallucination)) and `labels` (gold 0/1) on the test split |
| `test_metrics.json` | `metrics.auroc`, `metrics.aupr`, `metrics.ece`, `split`, `data_stats`, etc. |

### 4.2 Baseline (`outputs_failure_aware/`, `outputs_failure_aware_qa/`)

| Category | Files |
|----------|-------|
| **Required for comparison** | `predictions.json`, `test_metrics.json` |
| **Training / reproduction** | `best.pt`, `calibrator.pt`, tokenizer files (`tokenizer.json`, `vocab.txt`, …), `train_log.csv` |
| **Typical plots** | `calibration_curve.png`, `cav_curve.png` (may vary by run) |

Does **not** include Proposed-only interpretability exports (trust demos, signal agreement map, failure taxonomy JSON).

### 4.3 Proposed (`outputs_failure_aware_semantic_signal_reliability/`, `…_qa/`)

| Category | Files |
|----------|-------|
| **Required for comparison** | `predictions.json`, `test_metrics.json` |
| **Training / reproduction** | `best.pt`, `calibrator.pt`, tokenizer snapshot, `train_log.csv` |
| **Report / interpretability (Stage 3)** | `trust_demo_samples.json`, `failure_cases.json`, `signal_agreement_map.png` |
| **Diagnostics (typical)** | `roc_curve.png`, `pr_curve.png`, `calibration_curve.png`, `learning_curve.png`, `signal_importance.png`, `signal_correlation.png`, `feature_dist_*.png`, `confusion_matrix.png` |

### 4.4 SummaC (`outputs_summac_baseline/`, `outputs_summac_baseline_qa/`)

| File | Notes |
|------|-------|
| `predictions.json` | Consistency inverted to P(hallucination); same test indices as learned models |
| `test_metrics.json` | AUROC/AUPR/ECE on mapped scores |

No `best.pt` (not trained in this repo).

### 4.5 HaluEval judge (`outputs_halueval_baseline_qa/`)

| File | Notes |
|------|-------|
| `predictions.json` | From official `qa_gpt-3.5-turbo_result.json` via builder |
| `test_metrics.json` | QA test split only |

### 4.6 Main comparison (`outputs_comparison/`)

Produced by `compare_baseline.py` when Baseline + Proposed (+ optional SummaC) predictions exist:

| File | Purpose |
|------|---------|
| `roc_curve_comparison.png` | ROC: Baseline vs Proposed vs SummaC |
| `pr_curve_comparison.png` | PR curves |
| `calibration_comparison.png` | Reliability diagrams (binned) |
| `baseline_legend.json` | Model names and legend text for figures |

### 4.7 QA appendix comparison (`outputs_comparison_qa/`)

Same figure set as §4.6, produced by `qa_compare_baseline.py` (Baseline-QA, Proposed-QA, HaluEval judge, optional SummaC-QA).

---

## 5. End-to-end workflow (requirements traceability)

```text
Data/*.json
    │
    ├─► train_failure_aware.py ──────────────► outputs_failure_aware/
    ├─► train_failure_aware_semantic_signal_reliability.py
    │         └────────────────────────────► outputs_failure_aware_semantic_signal_reliability/
    ├─► run_summac_baseline.py ──────────────► outputs_summac_baseline/
    │
    └─► compare_baseline.py ─────────────────► outputs_comparison/

QA branch (dataset_type=qa):
    ├─► … ─► outputs_failure_aware_qa/
    ├─► … ─► outputs_failure_aware_semantic_signal_reliability_qa/
    ├─► run_summac_baseline.py ──────────────► outputs_summac_baseline_qa/
    ├─► build_halueval_baseline.py + halueval_baseline/*.json
    │         └────────────────────────────► outputs_halueval_baseline_qa/
    └─► qa_compare_baseline.py ────────────► outputs_comparison_qa/
```

**Report dependency (main chapter):** Proposed `test_metrics.json` + `outputs_comparison/*` figures, with Baseline and optional SummaC on identical test indices.

**Appendix dependency:** `outputs_comparison_qa/*` + `outputs_failure_aware_semantic_signal_reliability_qa/test_metrics.json` (and judge metrics in `outputs_halueval_baseline_qa/`).

---

## 6. Software environment (summary)

| Component | Requirement |
|-----------|-------------|
| Python | 3.10+ |
| pip | `pip install -r requirements.txt` |
| SummaC | included in `requirements.txt` (`summac>=0.0.4`) |
| Report DOCX | Node.js 18+ — `npm install` (see `package.json`, not pip) |
| Hardware | Local CPU / CUDA / Apple MPS; no cloud host |

---

## 7. Current committed reference metrics (all tasks, test split)

From `outputs_failure_aware_semantic_signal_reliability/test_metrics.json` (seed 42, `dataset_type=all`):

| Model | Source directory | AUROC | AUPR | ECE |
|-------|------------------|-------|------|-----|
| Proposed | `outputs_failure_aware_semantic_signal_reliability/` | 0.9247 | 0.9251 | 0.0355 |
| Baseline | `outputs_failure_aware/` | 0.8871 | 0.8894 | 0.0332 |
| SummaC | `outputs_summac_baseline/` | 0.6570 | 0.6895 | ~0.15 |

Do not change reported numbers without re-running the full pipeline with documented `Data/`, `--seed 42`, and the same code revision.

---

*Last aligned with: `train_failure_aware.py`, `train_failure_aware_semantic_signal_reliability.py`, `run_summac_baseline.py`, `compare_baseline.py`, `qa_compare_baseline.py`, and the `outputs_*` directories in the repository.*
