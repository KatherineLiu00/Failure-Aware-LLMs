# Failure-Aware LLMs: Learning to Predict and Signal Untrustworthy Outputs Across Tasks and Distributions

> Architecture details: [`README_SEMANTIC_SIGNAL_RELIABILITY.md`](README_SEMANTIC_SIGNAL_RELIABILITY.md).

---

## Table of contents

1. [Positioning and narrative](#1-positioning-and-narrative)
2. [Repository map](#2-repository-map) — see also [`PROJECT_REQUIREMENTS.md`](PROJECT_REQUIREMENTS.md)
3. [Data layout](#3-data-layout)
4. [Environment and hardware](#4-environment-and-hardware)
5. [Baseline: `train_failure_aware.py`](#5-baseline-train_failure_awarepy)
6. [Proposed: `train_failure_aware_semantic_signal_reliability.py`](#6-proposed-train_failure_aware_semantic_signal_reliabilitypy)
7. [SummaC external baseline](#7-summac-external-baseline)
8. [Main comparison: `compare_baseline.py`](#8-main-comparison-compare_baselinepy)
9. [Appendix comparison: `qa_compare_baseline.py`](#9-appendix-comparison-qa_compare_baselinepy)
10. [HaluEval judge baseline](#10-halueval-judge-baseline)
11. [Output files](#11-output-files)
12. [JSON and metrics semantics](#12-json-and-metrics-semantics)
13. [Calibration comparison figure](#13-calibration-comparison-figure)
14. [Citation and acknowledgements](#14-citation-and-acknowledgements)

---

## 1. Positioning and narrative

Many pipelines treat hallucination detection as **binary classification**. The **Proposed** method still outputs **P(hallucination)**, but emphasizes **risk-aware, interpretable** behaviour:

- **Calibrated probabilities** (Stage 3: temperature / Platt / isotonic / none)
- **SignalReliabilityAnalyzer** (Stage 2.5): explicit **agreement vs conflict** across semantic and uncertainty cues → **reliability_score**, **trust tier (HIGH / MEDIUM / LOW)**, and textual **reasoning**
- **Per-sample exports**: `trust_demo_samples.json`, `failure_cases.json`, `signal_agreement_map.png`, and additive per-signal attributions (SHAP-*style* heuristics, **not** official SHAP)

The **Baseline** (`train_failure_aware.py`) is the fair control: same encoder + uncertainty fusion, **without** the reliability branch, used for ablation in `compare_baseline.py`.

---

## 2. Repository map

| Path | Role |
|------|------|
| `train_failure_aware.py` | Baseline training (stages 1–3) → `outputs_failure_aware/` |
| `train_failure_aware_semantic_signal_reliability.py` | **Proposed** training + single-example **inference** → `outputs_failure_aware_semantic_signal_reliability/` |
| `PROJECT_REQUIREMENTS.md` | **Models, data, and output-directory requirements** (traceability matrix) |
| `compare_baseline.py` | **Main** PR / ROC / calibration comparison; optional SummaC |
| `qa_compare_baseline.py` | **Appendix**: QA-only + HaluEval GPT-3.5 judge + Baseline/Proposed (+ optional SummaC-QA) |
| `run_summac_baseline.py` | SummaC baseline (same `load_all_data` + `split_stratified` as training) |
| `halueval_baseline/build_halueval_baseline.py` | Convert official HaluEval judge results for comparison |
| `Data/*.json` | HaluEval four sub-tasks |
| `halueval_baseline/*.json` | Official QA judge output (e.g. `qa_gpt-3.5-turbo_result.json`) |

---

## 3. Data layout

### 3.1 Default data root

Training defaults to **`--data_dir Data`** (a `Data/` folder at the repo root).

### 3.2 HaluEval JSON files

| Sub-task | Filename | Typical scale |
|----------|----------|---------------|
| QA | `qa_data.json` | ~10k + 10k |
| Dialogue | `dialogue_data.json` | ~10k + 10k |
| Summarization | `summarization_data.json` | ~10k + 10k |
| General | `general_data.json` | ~3.7k correct + ~0.8k hallucinated |

With `dataset_type=all`, **only existing files** are loaded. If you see `Total samples: 0`, check paths and filenames.

### 3.3 Judge builder vs training `data_dir`

`build_halueval_baseline.py` defaults to `--data_dir HaluEval-Data`. If you train from `Data/`, pass **`--data_dir Data`** when building judge baselines so splits align.

---

## 4. Environment and hardware

- **Python**: 3.10+ (3.11 OK)
- **RAM**: ≥16 GB recommended (DistilBERT training, offline feature extraction, and SummaC on CPU are memory-heavy on smaller machines)
- **Install (pip):**

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

- **Packages:** `torch`, `transformers`, `scikit-learn`, `matplotlib`, `numpy`, `tqdm`, `summac` — see [`requirements.txt`](requirements.txt). Install `torch` first from [pytorch.org](https://pytorch.org/get-started/locally/) if the default wheel is wrong for your machine.
- **macOS**: default `num_workers=4`; use **`--num_workers 0`** on tokenizer fork issues
- **Apple Silicon (MPS)**: PyTorch MPS backend where available; reduce **`--batch_size`** or **`--max_length`** if OOM

---

## 5. Baseline: `train_failure_aware.py`

| Stage | Content |
|-------|---------|
| 1 | Encoder + binary head |
| 2 | Uncertainty proxies |
| 3 | Calibration + basic plots |

```bash
python train_failure_aware.py --stage 3 --data_dir Data --output_dir outputs_failure_aware
```

Before comparison, the output dir must contain **`predictions.json`** and **`test_metrics.json`**.

---

## 6. Proposed: `train_failure_aware_semantic_signal_reliability.py`

### 6.1 Modes (`--mode`)

| Value | Meaning |
|-------|---------|
| `train` (default) | Train + val + test; Stage ≥ 3 exports plots and JSON |
| `inference` | Load `best.pt` (+ optional `calibrator.pt`), run `--input` once |

### 6.2 Stages (`--stage`)

| Stage | Meaning |
|-------|---------|
| 1 | Semantic branch |
| 2 | + uncertainty branch |
| 3 | Full pipeline + calibration + interpretability exports |

Use **`--stage 3`** for delivery-style reproduction.

### 6.3 Training pipeline

1. `load_all_data` → prompts, labels, `data_stats`
2. `split_stratified` (default 80 / 10 / 10, `seed=42`)
3. Semantic + uncertainty features; **train-split statistics only** for normalization
4. Checkpoint by **val AUROC** → `best.pt`
5. Fit calibrator on val → `calibrator.pt`
6. Test evaluation → `test_metrics.json`, `predictions.json`

### 6.4 Common commands

```bash
# All tasks
python train_failure_aware_semantic_signal_reliability.py --stage 3 \
  --data_dir Data \
  --output_dir outputs_failure_aware_semantic_signal_reliability

# Single-example inference
python train_failure_aware_semantic_signal_reliability.py --mode inference \
  --model_path outputs_failure_aware_semantic_signal_reliability \
  --input "Knowledge: ... Question: ... Answer: ..."

# QA-only (appendix)
python train_failure_aware_semantic_signal_reliability.py --stage 3 \
  --dataset_type qa \
  --output_dir outputs_failure_aware_semantic_signal_reliability_qa
```

### 6.5 Key CLI defaults

| Arg | Default | Notes |
|-----|---------|-------|
| `--data_dir` | `Data` | HaluEval JSON root |
| `--dataset_type` | `all` | per-task: `qa`, `dialogue`, … |
| `--seed` | `42` | Match SummaC / judge when comparing |
| `--val_ratio` / `--test_ratio` | `0.1` / `0.1` | |
| `--lr` | `2e-5` | AdamW |
| `--max_length` | `256` | HaluEvalDataset truncation; SemanticAnalyzer uses 512 separately |
| `--calibration_method` | `temperature` | `platt` \| `isotonic` \| `none` |
| `--max_failure_cases_per_type` | `25` | `0` = all FP/FN |

---

## 7. SummaC external baseline

[SummaC](https://github.com/tingofurro/summac) is used as a **published external comparison baseline** (not trained in this repo). `run_summac_baseline.py` reuses **`load_all_data`** and **`split_stratified`** from the Proposed script so splits match your training run.

**Notes**

- Default `--data_dir` is `HaluEval-Data`—pass **`Data`** if that is your training root.
- Requires `summac` (included in `requirements.txt`)

```bash
# All tasks (third curve in main comparison)
python run_summac_baseline.py --data_dir Data --dataset_type all \
  --output_dir outputs_summac_baseline

# QA-only (appendix, optional)
python run_summac_baseline.py --data_dir Data --dataset_type qa \
  --output_dir outputs_summac_baseline_qa
```

---

## 8. Main comparison: `compare_baseline.py`

| Default path | Model |
|--------------|-------|
| `./outputs_failure_aware` | Baseline |
| `./outputs_failure_aware_semantic_signal_reliability` | Proposed |
| `./outputs_summac_baseline` | SummaC (optional) |
| `./outputs_comparison` | Output figures |

```bash
python compare_baseline.py
```

**Outputs:** `pr_curve_comparison.png`, `roc_curve_comparison.png`, `calibration_comparison.png`, `baseline_legend.json`, plus AUROC / AUPR / ECE in the terminal.

All `predictions.json` files must share the **same** `labels` length (`seed`, `val_ratio`, `test_ratio`, `data_dir` must align).

---

## 9. Appendix comparison: `qa_compare_baseline.py`

Compares Baseline, Proposed, **HaluEval official GPT-3.5-turbo judge**, and optional SummaC-QA on the **QA** sub-task.

| Default input dirs |
|--------------------|
| `outputs_failure_aware_qa` |
| `outputs_failure_aware_semantic_signal_reliability_qa` |
| `outputs_halueval_baseline_qa` |
| `outputs_summac_baseline_qa` (optional) |

Output: `outputs_comparison_qa/`. **Proposed-QA** test length is canonical.

```bash
python qa_compare_baseline.py
```

---

## 10. HaluEval judge baseline

Uses official [HaluEval](https://github.com/RUCAIBox/HaluEval) **QA + GPT-3.5-turbo** recognition results as an LLM-as-judge reference:

```bash
python halueval_baseline/build_halueval_baseline.py \
  --dataset_type qa \
  --result_file halueval_baseline/qa_gpt-3.5-turbo_result.json \
  --data_dir Data \
  --output_dir outputs_halueval_baseline_qa
```

Output must include **`predictions.json`** and **`test_metrics.json`**.

---

## 11. Output files

Per-directory minimum artefacts (and the main vs QA output map) are in [`PROJECT_REQUIREMENTS.md`](PROJECT_REQUIREMENTS.md).

### Proposed (`outputs_failure_aware_semantic_signal_reliability/`)

| Category | Files |
|----------|-------|
| Checkpoints | `best.pt`, `calibrator.pt`, tokenizer snapshot |
| Metrics | `test_metrics.json`, `predictions.json`, `train_log.csv` |
| Curves | `roc_curve.png`, `pr_curve.png`, `calibration_curve.png`, `cav_curve.png`, … |
| Signals | `signal_correlation.png`, `signal_importance.png`, `feature_dist_*.png` |
| Interpretability | `signal_agreement_map.png`, `trust_demo_samples.json`, `failure_cases.json` |

Baseline dirs usually **omit** the agreement map and extended JSON exports.

### Comparison dirs

- `outputs_comparison/` — all-task figures
- `outputs_comparison_qa/` — QA appendix figures

---

## 12. JSON and metrics semantics

### `test_metrics.json`

- `dataset`, `data_stats`, `split`, `metrics` (`auroc`, `aupr`, `ece`, `accuracy`, …)

### `predictions.json`

- `probs` — P(hallucination)
- `labels` — gold 0/1

### `failure_cases.json`

FP/FN with **`error_taxonomy`** (e.g. semantic failure / missing evidence / signal conflict) and **`signal_contributions`** (heuristic additive attribution, not certified SHAP).

---

## 13. Calibration comparison figure

**`calibration_comparison.png`** in `compare_baseline.py` / `qa_compare_baseline.py`:

- Quantile bins (default 5) to reduce empty bins under peaked scores
- Weighted isotonic regression on bin centres → smooth monotone curve; scatter = raw bin positive rates
- Legend **ECE** from each model’s `test_metrics.json` (independent of the binned plot)

> Baseline vs Proposed **ECE can be very close**—check numbers before claiming large calibration gains.

---

## 14. Citation and acknowledgements

### SummaC (external comparison baseline)

This project uses [SummaC](https://github.com/tingofurro/summac) as an **external comparison baseline** via `run_summac_baseline.py` on the same HaluEval splits. When reporting SummaC results, follow the [Apache-2.0 license](https://github.com/tingofurro/summac/blob/master/LICENSE) and cite:

```bibtex
@article{Laban2022SummaC,
  title={SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization},
  author={Philippe Laban and Tobias Schnabel and Paul N. Bennett and Marti A. Hearst},
  journal={Transactions of the Association for Computational Linguistics},
  year={2022},
  volume={10},
  pages={163--177}
}
```

### HaluEval (dataset and judge baseline)

This project uses [HaluEval](https://github.com/RUCAIBox/HaluEval) data (`Data/*.json`) to **train and evaluate** our Baseline and Proposed models. For the QA appendix, we also use the official **GPT-3.5-turbo** hallucination-recognition results (`halueval_baseline/qa_gpt-3.5-turbo_result.json`) as an **LLM-as-judge comparison baseline**. Please follow the [MIT license](https://github.com/RUCAIBox/HaluEval/blob/main/LICENSE) and cite:

```bibtex
@misc{HaluEval2023,
  author = {Junyi Li and Xiaoxue Cheng and Wayne Xin Zhao and Jian-Yun Nie and Ji-Rong Wen},
  title = {HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models},
  year = {2023},
  eprint = {2305.11747},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL}
}
```

---


