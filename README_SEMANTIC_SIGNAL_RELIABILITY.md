# Failure-Aware Hallucination Detector — Enhanced (`train_failure_aware_semantic_signal_reliability.py`)

An enhanced hallucination detection system built on **semantic features**, **uncertainty proxies**, and **signal reliability analysis**, with probability calibration and trust-tier outputs.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Three-Stage Architecture](#three-stage-architecture)
- [Differences from `train_failure_aware.py`](#differences-from-train_failure_awarepy)
- [Core Components](#core-components)
- [Feature Engineering Comparison](#feature-engineering-comparison)
- [Model Architecture Comparison](#model-architecture-comparison)
- [Full CLI Reference](#full-cli-reference)
- [Output Files](#output-files)

---

## Quick Start

### Training

```bash
# Stage 1: semantic features only
python train_failure_aware_semantic_signal_reliability.py --stage 1

# Stage 2: semantic + uncertainty features
python train_failure_aware_semantic_signal_reliability.py --stage 2

# Stage 3: full pipeline (semantic + uncertainty + calibration + signal reliability)
python train_failure_aware_semantic_signal_reliability.py --stage 3
```

### Inference

```bash
python train_failure_aware_semantic_signal_reliability.py \
    --mode inference \
    --model_path outputs_failure_aware_semantic_signal_reliability \
    --input "Knowledge: The Eiffel Tower is in Paris. Question: Where is the Eiffel Tower? Answer: Paris"
```

### Quick smoke test

```bash
# Run on 1,000 samples to verify the pipeline
python train_failure_aware_semantic_signal_reliability.py --stage 3 --max_samples 1000
```

---

## Three-Stage Architecture

### Stage 1: Semantic Analysis

**Goal**: Extract semantic alignment features between the answer and the provided knowledge.

**Idea**: Hallucinated answers are often inconsistent with the given knowledge; semantic cues can surface that mismatch.

**Pipeline**:

```
Input prompt
    │
    ├── Parse three components:
    │   ├── Knowledge: background context
    │   ├── Question: user query / dialogue turn
    │   └── Answer: candidate response to score
    │
    └── Four semantic features:
        │
        ├── ① semantic_similarity
        │   └── Cosine similarity between answer and knowledge (DistilBERT encoding)
        │
        ├── ② entity_overlap_ratio
        │   └── Share of answer entities that appear in knowledge
        │
        ├── ③ entity_coverage
        │   └── Share of knowledge entities that appear in the answer
        │
        └── ④ knowledge_overlap
            └── Keyword overlap ratio
```

**`SemanticAnalyzer` highlights**:

- DistilBERT text encoding
- Lightweight NER-style patterns (PERSON, LOCATION, DATE, NUMBER, ORGANIZATION)
- Word-level and entity-level coverage statistics
- Multiple HaluEval prompt layouts (QA, dialogue, summarisation, general)

### Stage 2: Uncertainty Features

**Goal**: Capture proxy uncertainty signals at answer level (no live generator logits required).

**Idea**: When cues suggest instability or weak grounding, hallucination risk tends to rise.

**Core signals** (also shown in the correlation heatmap):

| Feature | Family | Description | How it is computed |
|------|--------|-------------|-------------------|
| semantic_similarity | Semantic | Answer vs. knowledge cosine similarity | DistilBERT encoding |
| perplexity | Uncertainty | Perplexity proxy | Knowledge coverage + length + jitter |
| knowledge_overlap | Semantic | Keyword overlap ratio | Set intersection over answer keywords |
| token_entropy | Uncertainty | Token-level entropy proxy | Numeric density + sentence count + jitter |
| entity_overlap_ratio | Semantic | Entity overlap vs. answer entities | Regex entity matching |

**Note**: The heatmap focuses on the five signals above. Additional fields (`answer_length`, `answer_char_length`, `avg_confidence`, `sequence_entropy`) are fed to the classifier but omitted from the heatmap for readability.

**Auxiliary uncertainty fields**:

| Feature | Description | How it is computed |
|------|-------------|-------------------|
| answer_length | Answer length | Token/word count |
| answer_char_length | Character length | Character count |
| avg_confidence | Average confidence proxy | Tied to knowledge overlap |
| sequence_entropy | Sequence entropy | token_entropy × answer_length |


### Stage 2.5: Signal Reliability Analysis

**Goal**: Judge when heterogeneous cues agree, detect cross-family conflicts, and emit a composite reliability score.

**Idea**: A single scalar or a single cue family can mislead; deployment needs explicit conflict reporting.

**Pipeline**:

```
Input features (semantic + uncertainty)
    │
    ├── 1. Quantise each cue to {high, moderate, low}
    │   ├── semantic_similarity: ≥0.7 → high, ≤0.3 → low
    │   ├── perplexity: ≤10 → high, ≥30 → low (inverted)
    │   ├── token_entropy: ≤0.5 → high, ≥2.0 → low (inverted)
    │   ├── knowledge_overlap: ≥0.6 → high, ≤0.2 → low
    │   └── entity_overlap_ratio: ≥0.5 → high, ≤0.15 → low
    │
    ├── 2. Consistency score
    │   └── High consistency = multiple cues point the same way
    │
    ├── 3. Conflict detection
    │   ├── Semantic similarity vs. perplexity
    │   ├── Knowledge overlap vs. semantic similarity
    │   └── Token entropy vs. perplexity
    │
    └── 4. Reliability score
        ├── reliability_score = weighted consistency + agreement − conflict penalty
        └── trust_level: HIGH / MEDIUM / LOW
```

**`SignalReliabilityAnalyzer` class**:

```python
class SignalReliabilityAnalyzer:
    def __init__(self,
                 consistency_weight=0.4,
                 conflict_penalty=0.3,
                 confidence_agreement_weight=0.3):
        ...

    def analyze(self, features, model_confidence) -> Dict:
        """
        Returns:
        {
            'reliability_score': 0.0-1.0,      # composite reliability
            'trust_level': 'HIGH/MEDIUM/LOW',  # routing tier
            'consistent_signals': [...],        # aligned cues
            'conflicting_signals': [...],       # disagreeing cues
            'conflicts': [...],                 # human-readable conflict strings
            'adjusted_confidence': 0.0-1.0    # optional nudge (export only)
        }
        """
```

**Why Stage 2.5 exists**:

1. **Cues can mislead**: High semantic similarity does not always mean a correct answer.
2. **Cues can conflict**: Semantic and uncertainty families may disagree.
3. **Routing needs structure**: Trust tiers complement calibrated P(hallucination).

**Conflict patterns**:

| Conflict type | Description |
|---------------|-------------|
| Semantic + perplexity | High similarity but high perplexity proxy |
| Knowledge + semantic | Large gap between knowledge overlap and semantic similarity (>0.4) |
| Entropy + perplexity | High perplexity but low entropy proxy, or the reverse |

**Confidence adjustment (export field)**:

```
adjusted_confidence = model_confidence × reliability_score
```

**Placement**: After Stage 2 fusion, before Stage 3 calibration (does not change ranking logits used for AUROC).


### Stage 3: Calibration and Visualisation

**Goal**: Calibrate classifier probabilities and write diagnostic plots.

**Calibration methods**:

| Method | Description |
|--------|-------------|
| Temperature scaling | Single temperature on validation logits |
| Platt scaling | Linear map on logits |
| Isotonic regression | Non-parametric monotonic map |

**Plots written under the output directory**:
- CAV-style curve (coverage vs. accuracy)
- Reliability / calibration diagram (confidence vs. accuracy)
- ROC curve
- Precision–recall curve
- Confusion matrix heatmap
- Signal correlation heatmap

---

## Differences from `train_failure_aware.py`

### Overview

| Aspect | `train_failure_aware.py` | `train_failure_aware_semantic_signal_reliability.py` |
|--------|--------------------------|------------------------------------------------------|
| Stage 1 | Basic binary classifier | **Semantic-aware** classifier (`SemanticAnalyzer`) |
| Semantic features | None | Four semantic scalars |
| Fusion | Text embedding only | **Text + semantic + uncertainty** |
| Uncertainty proxies | Coupled length buckets | **Decoupled formulas + jitter** |
| Signal reliability | None | **`SignalReliabilityAnalyzer`** |
| Console / JSON exports | Basic | **Trust tier + conflict strings** |

### 1. Stage 1 upgrade: from text-only to semantic-aware

**Legacy**:

```python
class HallucinationPredictor(nn.Module):
    """Basic binary classifier."""
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] only
        return self.head(self.dropout(pooled))
```

**Enhanced**:

```python
class SemanticAwareHallucinationPredictor(nn.Module):
    """Semantic-aware classifier."""
    def forward(self, input_ids, attention_mask,
                semantic_similarity, entity_overlap_ratio,
                knowledge_overlap, entity_coverage):
        # 1. Text embedding
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = outputs.last_hidden_state[:, 0, :]

        # 2. Semantic feature projection
        semantic_feats = torch.stack([semantic_similarity, entity_overlap_ratio,
                                       knowledge_overlap, entity_coverage], dim=1)
        semantic_emb = self.semantic_proj(semantic_feats)

        # 3. Fusion
        combined = torch.cat([text_emb, semantic_emb], dim=1)
        fused = self.fusion_layer(combined)
        return self.head(self.dropout(fused))
```

**Benefits**:
- Semantic cues enter the decision boundary explicitly.
- Easier to audit feature contributions.
- More sensitive to knowledge–answer mismatches HaluEval adversarially preserves at high lexical overlap.

### 2. Improved uncertainty proxies

**Legacy issue**: `perplexity` and `token_entropy` were tied to the same `answer_length` buckets, yielding near-perfect correlation (≈1.0).

**Legacy code**:

```python
# Perplexity proxy (length buckets only)
if features['answer_length'] <= 3:
    features['perplexity'] = 1.5
    features['token_entropy'] = 0.5  # fixed pairing → artificial correlation
elif features['answer_length'] <= 10:
    features['perplexity'] = 3.0
    features['token_entropy'] = 1.2
else:
    features['perplexity'] = 5.0
    features['token_entropy'] = 2.0
```

**Enhanced code**:

```python
# Perplexity: driven by knowledge coverage
knowledge_cov = features.get('knowledge_overlap', 0.5)
base_perplexity = 1.0 + (1.0 - knowledge_cov) * 8.0
if features['answer_length'] > 10:
    base_perplexity *= 1.2
features['perplexity'] = base_perplexity + random.uniform(-0.5, 0.5)

# Token entropy: numeric density + sentence count (different sources)
num_density = features.get('numeric_density', 0)
sent_count = features.get('sentence_count', 1)
base_entropy = 0.5 + num_density * 2.0
base_entropy += min(sent_count / 10.0, 1.0)
features['token_entropy'] = base_entropy + random.uniform(-0.3, 0.3)
```

**Benefits**:
- Perplexity and token entropy use **different inputs**.
- Jitter mimics sampling variability in real decoders.
- Correlation structure is more realistic for Stage 2.5 conflict rules.

### 3. New: `SignalReliabilityAnalyzer`

```python
class SignalReliabilityAnalyzer:
    """
    Post-hoc reliability layer:
    1. Consistency over quantised cues
    2. Conflict detection across families
    3. Composite reliability_score
    4. Optional adjusted_confidence for exports
    """
```

**Analysis flow** (same thresholds as in the paper’s Table of SRA thresholds):

```
Input features (semantic + uncertainty)
    │
    ├── 1. Quantise cues
    ├── 2. Consistency
    ├── 3. Conflicts (semantic vs. uncertainty families)
    └── 4. trust_level: HIGH / MEDIUM / LOW
```

**Example console output**:

```
============================================================
Answer: ZeniMax Online Studios
============================================================
P(hallucination) = 0.006
Final confidence = 100.0% [after reliability layer]
  └─ Model confidence = 99.4% (calibrated)
  └─ Reliability score = 92.0%
Trust level: HIGH
✓ High reliability (92%); cues aligned; displayed confidence 100.0%

--- Signal reliability details ---
Aligned cues: semantic_similarity (high), perplexity (low),
              knowledge_overlap (high), entity_overlap (high)
============================================================
```

### 4. Stage 2 model upgrade

**Legacy**: `EnhancedHallucinationPredictor` (text + uncertainty only).

**Enhanced**: `EnhancedHallucinationPredictor` (text + semantic + uncertainty).

```python
class EnhancedHallucinationPredictor(nn.Module):
    def __init__(self, encoder, hidden_size,
                 n_semantic_features=4,
                 n_uncertainty_features=6,
                 dropout=0.1):
        # Semantic branch
        self.semantic_proj = nn.Sequential(
            nn.Linear(n_semantic_features, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, hidden_size // 8),
        )

        # Uncertainty branch
        self.uncertainty_proj = nn.Sequential(...)

        # Fusion: text + semantic + uncertainty
        fusion_dim = hidden_size + hidden_size // 8 + hidden_size // 8
```

---

## Core Components

### `SemanticAnalyzer`

Extracts semantic features from a HaluEval-style prompt string.

```python
class SemanticAnalyzer:
    def compute_semantic_similarity(self, text1, text2) -> float:
        """Cosine similarity of encodings."""

    def extract_entities(self, text) -> List[str]:
        """Lightweight entity extraction."""

    def compute_entity_overlap(self, answer, knowledge) -> Dict:
        """Entity overlap statistics."""

    def compute_knowledge_coverage(self, answer, knowledge) -> Dict:
        """Keyword / coverage statistics."""
```

### `SignalReliabilityAnalyzer`

Scores cue agreement and emits trust metadata.

```python
class SignalReliabilityAnalyzer:
    def analyze(self, features, model_confidence) -> Dict:
        """
        Returns:
        {
            'reliability_score': 0.0-1.0,
            'trust_level': 'HIGH/MEDIUM/LOW',
            'consistent_signals': [...],
            'conflicting_signals': [...],
            'adjusted_confidence': 0.0-1.0
        }
        """
```

---

## Feature Engineering Comparison

### Stage 1 features

| Feature | `train_failure_aware.py` | `train_failure_aware_semantic_signal_reliability.py` |
|------|--------------------------|------------------------------------------------------|
| Text embedding | DistilBERT [CLS] | DistilBERT [CLS] + optional mean pooling |
| semantic_similarity | — | Sentence-level cosine on K vs. A |
| entity_overlap_ratio | — | Entity set overlap |
| knowledge_overlap | — | Keyword overlap |
| entity_coverage | — | Knowledge entities covered in answer |

### Stage 2 features

| Feature | `train_failure_aware.py` | `train_failure_aware_semantic_signal_reliability.py` |
|------|--------------------------|------------------------------------------------------|
| perplexity | Fixed length buckets | knowledge_overlap + jitter |
| token_entropy | Fixed length buckets | numeric_density + sentences + jitter |
| answer_length | Yes | Yes |
| answer_char_length | Yes | Yes |
| avg_confidence | Yes | Yes |
| sequence_entropy | Yes | Yes |

---

## Model Architecture Comparison

### Stage 1 models

| Aspect | `train_failure_aware.py` | `train_failure_aware_semantic_signal_reliability.py` |
|------|--------------------------|------------------------------------------------------|
| Class | `HallucinationPredictor` | `SemanticAwareHallucinationPredictor` |
| Inputs | Text only | Text + 4 semantic scalars |
| Fusion | None | Project semantic vector + concatenate |
| Interpretability | Low | Higher (explicit features) |

### Stage 2 models

| Aspect | `train_failure_aware.py` | `train_failure_aware_semantic_signal_reliability.py` |
|------|--------------------------|------------------------------------------------------|
| Class | `EnhancedHallucinationPredictor` | `EnhancedHallucinationPredictor` (extended) |
| Inputs | Text + uncertainty | Text + semantic + uncertainty |
| Fusion width | hidden + hidden/4 | hidden + hidden/8 + hidden/8 |

---

## Full CLI Reference

```bash
python train_failure_aware_semantic_signal_reliability.py \
    --stage 3                  # Training stage: 1, 2, or 3
    --mode train               # train or inference
    --epochs 10                # Epochs (default in script may be 3)
    --batch_size 16            # Batch size
    --lr 2e-5                  # Learning rate
    --max_length 256           # Max sequence length (CLI default; shipped best.pt uses 256)
    --dropout 0.1              # Dropout
    --calibration_method isotonic  # temperature | platt | isotonic | none
    --use_mean_pooling         # Mean pooling instead of [CLS] for full prompt encoding
    --max_samples 10000        # Cap dataset size for quick runs
    --seed 42                  # Random seed

    # Inference
    --model_path outputs_failure_aware_semantic_signal_reliability
    --input "Your HaluEval-style prompt here"
```

---

## Output Files

Typical layout after training (`--stage 3`):

```
outputs_failure_aware_semantic_signal_reliability/
├── best.pt                     # Best checkpoint (validation AUROC)
├── calibrator.pt               # Temperature / Platt / isotonic calibrator
├── test_metrics.json           # AUROC, AUPR, accuracy, ECE
├── predictions.json            # Per-sample probs, labels, trust fields
├── train_log.csv               # Epoch-wise train/val metrics
├── failure_cases.json          # Error taxonomy export
├── trust_demo_samples.json     # Trust-tier demo rows
│
├── cav_curve.png               # Coverage vs. accuracy
├── calibration_curve.png       # Reliability diagram
├── signal_correlation.png      # Feature correlation heatmap
└── reliability_distribution.png  # Reliability score histogram (when generated)
```

Older runs may also contain `model.pt`, `metrics.json`, or `calibration_params.json`; the canonical names above match the current training script.

---

## Summary: Why use the enhanced script?

1. **Explicit semantics**: Grounding cues are injected directly, not only via implicit [CLS] representation.
2. **Decoupled uncertainty proxies**: Perplexity and entropy no longer move in lockstep.
3. **Deployable trust layer**: `SignalReliabilityAnalyzer` surfaces conflicts and HIGH/MEDIUM/LOW routing.
4. **Auditable exports**: JSON artefacts list cues, conflicts, and heuristic `signal_contributions`.

### When to use which script

| Scenario | Recommended script |
|------|-------------------|
| Quick baseline / minimal deps | `train_failure_aware.py` |
| Production-style verifier / paper numbers | `train_failure_aware_semantic_signal_reliability.py` |
| Explainability / trust tiers | `train_failure_aware_semantic_signal_reliability.py` |
| Feature or ablation studies | `train_failure_aware_semantic_signal_reliability.py` |
