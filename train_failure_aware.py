"""
Failure-Aware hallucination detector — integrated Stages 1+2+3.
Built on a Transformer encoder (DistilBERT).

Features:
- Stage 1: Binary classification (ECE, AUROC, AUPR, CAV)
- Stage 2: Uncertainty features (perplexity, entropy, self-consistency proxy, answer length)
- Stage 3: Calibration (temperature / Platt / isotonic) + trust signal + CAV plots

Usage:
    # Stage 1 (default)
    python train_failure_aware.py --stage 1

    # Stage 2 (extra uncertainty features)
    python train_failure_aware.py --stage 2

    # Stage 3 (full pipeline with calibration and plots)
    python train_failure_aware.py --stage 3

    # Inference + trust signal
    python train_failure_aware.py --mode inference --input "Knowledge: ... Question: ... Answer: ..."
"""

import argparse
import csv
import json
import math
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# ============================================================================
# ========== Stage 1: Binary classification core (from legacy train_hallucination_detector.py)
# ============================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class HaluEvalDataset(Dataset):
    """Dataset shared by Stage 1 and Stage 2."""
    def __init__(self, prompts: List[str], labels: List[int], tokenizer, max_length: int,
                 uncertainty_features: Optional[List[Dict]] = None):
        self.items = []
        for i, (prompt, label) in enumerate(zip(prompts, labels)):
            item = {"prompt": prompt, "label": label}
            if uncertainty_features is not None:
                item["uncertainty"] = uncertainty_features[i]
            self.items.append(item)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        enc = self.tokenizer(item["prompt"], truncation=True, max_length=self.max_length, return_tensors=None)
        result = {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "label": torch.tensor(item["label"], dtype=torch.float32),
        }
        if "uncertainty" in item:
            result["uncertainty"] = {k: torch.tensor(v, dtype=torch.float32) for k, v in item["uncertainty"].items()}
        return result


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    label: torch.Tensor
    # Stage 2: uncertainty features
    perplexity: Optional[torch.Tensor] = None
    token_entropy: Optional[torch.Tensor] = None
    answer_length: Optional[torch.Tensor] = None
    answer_char_length: Optional[torch.Tensor] = None
    avg_confidence: Optional[torch.Tensor] = None
    sequence_entropy: Optional[torch.Tensor] = None


def collate_fn(features: List[Dict]) -> Batch:
    input_ids = [f["input_ids"] for f in features]
    attention_mask = [f["attention_mask"] for f in features]

    max_len = max(x.size(0) for x in input_ids)
    padded_ids = torch.zeros((len(features), max_len), dtype=torch.long)
    padded_mask = torch.zeros((len(features), max_len), dtype=torch.long)

    for i in range(len(features)):
        seq_len = input_ids[i].size(0)
        padded_ids[i, :seq_len] = input_ids[i]
        padded_mask[i, :seq_len] = attention_mask[i]

    batch = Batch(
        input_ids=padded_ids,
        attention_mask=padded_mask,
        label=torch.stack([f["label"] for f in features]),
    )

    # Stage 2: attach uncertainty tensors when present
    if "uncertainty" in features[0]:
        batch.perplexity = torch.stack([f["uncertainty"].get("perplexity", 0) for f in features])
        batch.token_entropy = torch.stack([f["uncertainty"].get("token_entropy", 0) for f in features])
        batch.answer_length = torch.stack([f["uncertainty"].get("answer_length", 0) for f in features])
        batch.answer_char_length = torch.stack([f["uncertainty"].get("answer_char_length", 0) for f in features])
        batch.avg_confidence = torch.stack([f["uncertainty"].get("avg_confidence", 0.5) for f in features])
        batch.sequence_entropy = torch.stack([f["uncertainty"].get("sequence_entropy", 0) for f in features])

    return batch


# ============================================================================
# ========== Stage 1: Model definition
# ============================================================================

class HallucinationPredictor(nn.Module):
    """Stage 1 model — baseline binary classifier."""
    def __init__(self, encoder: AutoModel, hidden_size: int, dropout: float = 0.1,
                 use_mean_pooling: bool = False):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)
        self.use_mean_pooling = use_mean_pooling

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state

        if self.use_mean_pooling:
            mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = last_hidden[:, 0, :]

        pooled = self.dropout(pooled)
        return self.head(pooled).squeeze(-1)


# ============================================================================
# ========== Stage 2: Enhanced model + uncertainty features
# ============================================================================

class EnhancedHallucinationPredictor(nn.Module):
    """
    Stage 2 model — enhanced with fused uncertainty features.

    Inputs: text embedding + uncertainty features.
    Output: P(hallucination).
    """
    def __init__(self, encoder: AutoModel, hidden_size: int,
                 n_uncertainty_features: int = 6, dropout: float = 0.1,
                 use_mean_pooling: bool = False):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)

        # Uncertainty feature projection
        self.uncertainty_proj = nn.Sequential(
            nn.Linear(n_uncertainty_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, hidden_size // 4),
        )

        # Fusion layers
        fusion_dim = hidden_size + hidden_size // 4
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
        )

        self.head = nn.Linear(hidden_size // 2, 1)
        self.use_mean_pooling = use_mean_pooling

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                perplexity: torch.Tensor, token_entropy: torch.Tensor,
                answer_length: torch.Tensor, answer_char_length: torch.Tensor,
                avg_confidence: torch.Tensor, sequence_entropy: torch.Tensor) -> torch.Tensor:
        # Text embedding
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state

        if self.use_mean_pooling:
            mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
            text_emb = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            text_emb = last_hidden[:, 0, :]

        # Uncertainty features
        uncertainty_feats = torch.stack([
            perplexity, token_entropy,
            answer_length / 1000.0, answer_char_length / 10000.0,
            avg_confidence, sequence_entropy / 10.0
        ], dim=1)

        uncertainty_emb = self.uncertainty_proj(uncertainty_feats)

        # Fuse text + uncertainty
        combined = torch.cat([text_emb, uncertainty_emb], dim=1)
        fused = self.fusion_layer(combined)

        return self.head(self.dropout(fused)).squeeze(-1)


# ============================================================================
# ========== Stage 2: Uncertainty feature computation
# ============================================================================

def compute_uncertainty_features(prompt: str) -> Dict[str, float]:
    """
    Heuristic uncertainty features for the answer span.

    Note: in production you can use LLM log-prob APIs for real perplexity / entropy;
    here we estimate from shallow text statistics.
    """
    features = {}
    answer_start = prompt.lower().find("answer:")

    if answer_start != -1:
        answer = prompt[answer_start + 7:].strip()
        features['answer_length'] = len(answer.split())
        features['answer_char_length'] = len(answer)

        # Numeric density
        numbers = len(re.findall(r'\d+', answer))
        words = answer.split()
        features['numeric_density'] = numbers / max(len(words), 1)

        # Sentence count
        features['sentence_count'] = max(answer.count('.') + answer.count('!') + answer.count('?'), 1)

        # Knowledge overlap proxy
        knowledge_start = prompt.lower().find("knowledge:")
        if knowledge_start != -1:
            knowledge_end = prompt.lower().find("\n", knowledge_start + 10)
            if knowledge_end == -1:
                knowledge_end = len(prompt)
            knowledge = prompt[knowledge_start + 10:knowledge_end].lower()
            answer_words = set(answer.lower().split())
            knowledge_words = set(knowledge.split())
            overlap = len(answer_words & knowledge_words) / max(len(answer_words), 1)
            features['knowledge_overlap'] = overlap
        else:
            features['knowledge_overlap'] = 0.5
    else:
        features['answer_length'] = 0
        features['answer_char_length'] = 0
        features['numeric_density'] = 0
        features['sentence_count'] = 1
        features['knowledge_overlap'] = 0.5

    # Perplexity proxy (from answer length + knowledge overlap)
    if features['answer_length'] <= 3:
        features['perplexity'] = 1.5
        features['token_entropy'] = 0.5
    elif features['answer_length'] <= 10:
        features['perplexity'] = 3.0
        features['token_entropy'] = 1.2
    else:
        features['perplexity'] = 5.0
        features['token_entropy'] = 2.0

    # Confidence proxy
    features['avg_confidence'] = features['knowledge_overlap']
    features['sequence_entropy'] = features['token_entropy'] * features['answer_length']

    return features


def normalize_features(train_features: List[Dict], target_features: List[Dict]) -> Tuple[List[Dict], Dict]:
    """Normalize features using training-set statistics."""
    keys = ['perplexity', 'token_entropy', 'answer_length', 'answer_char_length',
            'avg_confidence', 'sequence_entropy']

    stats = {}
    for key in keys:
        values = [f[key] for f in train_features]
        stats[key] = {'mean': np.mean(values), 'std': np.std(values) + 1e-8}

    normalized = []
    for f in target_features:
        norm_f = {}
        for key in keys:
            norm_f[key] = (f[key] - stats[key]['mean']) / stats[key]['std']
        normalized.append(norm_f)

    return normalized, stats


# ============================================================================
# ========== Stage 3: Calibration
# ============================================================================

class TemperatureScaling(nn.Module):
    """Temperature scaling."""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.01)


class PlattScaling(nn.Module):
    """Platt scaling."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits * self.weight + self.bias


class IsotonicCalibrator:
    """Isotonic regression calibrator."""
    def __init__(self):
        self.calibrator = None

    def fit(self, probs: np.ndarray, labels: np.ndarray):
        self.calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        self.calibrator.fit(probs, labels)

    def predict(self, probs: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            return probs
        return self.calibrator.predict(probs)


# ============================================================================
# ========== Stage 3: Trust signal
# ============================================================================

def generate_trust_signal(confidence: float,
                         threshold_high: float = 0.85,
                         threshold_low: float = 0.35) -> Tuple[str, str]:
    """Produce discrete trust label + human-readable message."""
    if confidence >= threshold_high:
        return "HIGH", f"✓ High confidence ({confidence:.1%}); model is confident in the answer."
    elif confidence >= threshold_low:
        return "MEDIUM", f"⚠ Medium confidence ({confidence:.1%}); verify when possible."
    else:
        return "LOW", f"✗ Low confidence ({confidence:.1%}); treat with caution."


def format_answer_output(answer: str, confidence: float,
                         p_hallucination: float,
                         is_calibrated: bool = False) -> str:
    """Pretty-print answer text plus trust summary."""
    trust_level, trust_msg = generate_trust_signal(confidence)

    calibration_note = ""
    if is_calibrated:
        calibration_note = f" [post-calibration confidence]"

    output = f"""
{'='*60}
Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}
{'='*60}
P(hallucination) = {p_hallucination:.3f}
Confidence = {confidence:.1%}{calibration_note}
Trust level: {trust_level}
{trust_msg}
{'='*60}
"""
    return output


def _extract_demo_answer(prompt: str) -> Optional[str]:
    """Answer extraction aligned with the semantic script for multi-task demos."""
    for pattern in (
        r"Answer:\s*(.+?)(?:\n|$)",
        r"Response:\s*(.+?)(?:\n|$)",
        r"Summary:\s*(.+?)(?:\n|$)",
    ):
        m = re.search(pattern, prompt, re.DOTALL | re.IGNORECASE)
        if m:
            text = m.group(1).strip()
            if text:
                return text
    return None


def select_demo_indices_by_p_hallucination_quantiles(
    test_prompts: List[str],
    raw_probs: List[float],
    n_samples: int = 5,
    min_answer_chars: int = 3,
) -> List[int]:
    """Stratified sampling over P(hallucination) in [0,1] (equal quantiles)."""
    arr = np.asarray(raw_probs, dtype=np.float64)
    valid_indices: List[int] = []
    for idx, prompt in enumerate(test_prompts):
        ans = _extract_demo_answer(prompt)
        if ans is not None and len(ans) >= min_answer_chars:
            valid_indices.append(idx)
    if len(valid_indices) < n_samples:
        valid_indices = list(range(len(test_prompts)))

    valid_probs = arr[valid_indices]
    edges = np.linspace(0.0, 1.0, n_samples + 1)
    selected: List[int] = []
    for j in range(n_samples):
        low, high = float(edges[j]), float(edges[j + 1])
        last = j == n_samples - 1
        if last:
            mask = (valid_probs >= low) & (valid_probs <= high)
        else:
            mask = (valid_probs >= low) & (valid_probs < high)
        in_local = np.flatnonzero(mask)
        if in_local.size > 0:
            pick = int(in_local[in_local.size // 2])
            selected.append(valid_indices[pick])
        else:
            mid = 0.5 * (low + high)
            pick = int(np.argmin(np.abs(valid_probs - mid)))
            selected.append(valid_indices[pick])
    return selected


def _spread_pick_indices(sorted_inds: List[int], k: int) -> List[int]:
    """Pick up to k indices spread across sorted_inds (by underlying score order)."""
    if not sorted_inds:
        return []
    if len(sorted_inds) <= k:
        return sorted_inds[:]
    positions = [round(j * (len(sorted_inds) - 1) / max(k - 1, 1)) for j in range(k)]
    out: List[int] = []
    seen = set()
    for p in positions:
        idx = sorted_inds[int(p)]
        if idx not in seen:
            out.append(idx)
            seen.add(idx)
    for idx in sorted_inds:
        if len(out) >= k:
            break
        if idx not in seen:
            out.append(idx)
            seen.add(idx)
    return out[:k]


def select_demo_indices_by_trust_level(
    test_prompts: List[str],
    raw_probs: List[float],
    trust_fn: Callable[[int], str],
    n_per_level: int = 2,
    min_answer_chars: int = 3,
) -> List[int]:
    """Pick examples per trust tier (HIGH / MEDIUM / LOW), spread within tier by P(hallucination)."""
    arr = np.asarray(raw_probs, dtype=np.float64)
    valid_indices: List[int] = []
    for idx, prompt in enumerate(test_prompts):
        ans = _extract_demo_answer(prompt)
        if ans is not None and len(ans) >= min_answer_chars:
            valid_indices.append(idx)
    if len(valid_indices) < 1:
        valid_indices = list(range(len(test_prompts)))

    buckets: Dict[str, List[int]] = {"HIGH": [], "MEDIUM": [], "LOW": []}
    for i in valid_indices:
        lvl = trust_fn(i)
        if lvl not in buckets:
            lvl = "MEDIUM"
        buckets[lvl].append(i)

    selected: List[int] = []
    for lvl in ("HIGH", "MEDIUM", "LOW"):
        inds = buckets[lvl]
        if not inds:
            continue
        inds_sorted = sorted(inds, key=lambda j: float(arr[j]))
        selected.extend(_spread_pick_indices(inds_sorted, n_per_level))
    return selected


# ============================================================================
# ========== Metrics
# ============================================================================

def compute_ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def compute_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Aggregate scalar metrics."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    metrics = {}

    # AUROC / AUPR
    if len(np.unique(labels)) > 1:
        metrics['auroc'] = float(roc_auc_score(labels, probs))
        metrics['aupr'] = float(average_precision_score(labels, probs))
    else:
        metrics['auroc'] = float('nan')
        metrics['aupr'] = float('nan')

    # ECE
    metrics['ece'] = compute_ece(probs, labels)
    metrics['ece_10'] = compute_ece(probs, labels, 10)

    # Accuracy @ confidence thresholds
    correct = 1 - labels
    for thresh in [0.1, 0.2, 0.3]:
        mask = (probs <= thresh) | (probs >= (1 - thresh))
        metrics[f'acc@{thresh}_conf'] = float(correct[mask].mean()) if mask.any() else float('nan')

    return metrics


# ============================================================================
# ========== Inference helpers
# ============================================================================

@torch.no_grad()
def predict(model: nn.Module, batch: Batch, device: torch.device,
            stage: int = 1, calibrator=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass returning probabilities (+ logits)."""
    model.eval()

    kwargs = {
        'input_ids': batch.input_ids.to(device),
        'attention_mask': batch.attention_mask.to(device),
    }

    stage_int = int(stage)
    if stage_int >= 2:
        kwargs.update({
            'perplexity': batch.perplexity.to(device),
            'token_entropy': batch.token_entropy.to(device),
            'answer_length': batch.answer_length.to(device),
            'answer_char_length': batch.answer_char_length.to(device),
            'avg_confidence': batch.avg_confidence.to(device),
            'sequence_entropy': batch.sequence_entropy.to(device),
        })

    logits = model(**kwargs)

    # Optional calibration
    if calibrator is not None:
        if isinstance(calibrator, (TemperatureScaling, PlattScaling)):
            calibrated_logits = calibrator(logits)
        else:
            probs = torch.sigmoid(logits)
            calibrated_probs = calibrator.predict(probs.cpu().numpy())
            calibrated_logits = torch.tensor(calibrated_probs, device=device)
            return torch.sigmoid(calibrated_logits), logits

        return torch.sigmoid(calibrated_logits), logits

    return torch.sigmoid(logits), logits


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device,
              stage: int = 1, calibrator=None) -> Dict:
    """Evaluate model on a dataloader."""
    all_probs, all_labels, all_logits = [], [], []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        probs, logits = predict(model, batch, device, stage, calibrator)
        all_probs.extend(probs.cpu().numpy().tolist())
        all_logits.extend(logits.cpu().numpy().tolist())
        all_labels.extend(batch.label.numpy().tolist())

    metrics = compute_metrics(np.array(all_probs), np.array(all_labels))
    metrics['raw_probs'] = all_probs
    metrics['raw_logits'] = all_logits
    metrics['labels'] = all_labels

    return metrics


# ============================================================================
# ========== Stage 3: Visualization
# ============================================================================

def plot_cav_curve(model: nn.Module, dataloader: DataLoader, device: torch.device,
                    output_path: str, stage: int = 1, calibrator=None):
    """Coverage-vs-accuracy (CAV) plot."""
    model.eval()
    all_probs, all_labels = [], []

    for batch in tqdm(dataloader, desc="CAV", leave=False):
        probs, _ = predict(model, batch, device, stage, calibrator)
        all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(batch.label.numpy().tolist())

    probs_np = np.array(all_probs)
    correct = 1 - np.array(all_labels)

    thresholds = np.linspace(0, 1, 50)
    coverages, accuracies = [], []

    for tau in thresholds:
        mask = probs_np <= tau
        coverage = float(mask.mean())
        acc = float(correct[mask].mean()) if mask.any() else float('nan')
        coverages.append(coverage)
        accuracies.append(acc)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    valid = ~np.isnan(accuracies)
    ax1.plot(np.array(coverages)[valid], np.array(accuracies)[valid], 'b-', linewidth=2)
    ax1.axhline(y=0.9, color='g', linestyle='--', alpha=0.5, label='90% acc')
    ax1.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='80% acc')
    ax1.set_xlabel('Coverage')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('CAV Curve (Coverage vs Accuracy)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.05])

    ax2.bar(range(len(thresholds)), [1 - c for c in coverages], alpha=0.7)
    ax2.set_xlabel('Threshold Index')
    ax2.set_ylabel('Abstention Rate')
    ax2.set_title('Abstention Rate by Threshold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"CAV curve saved to {output_path}")


def plot_calibration_curve(probs: np.ndarray, labels: np.ndarray,
                            output_path: str):
    """Reliability diagram."""
    fig, ax = plt.subplots(figsize=(8, 8))

    n_bins = 15
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers, bin_accs = [], []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        if mask.sum() > 0:
            bin_centers.append((lo + hi) / 2)
            bin_accs.append(labels[mask].mean())

    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    ax.scatter(bin_centers, bin_accs, s=100, c='blue', label='Model')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Calibration Curve (ECE={compute_ece(probs, labels):.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Calibration curve saved to {output_path}")


# ============================================================================
# ========== Data loading (HaluEval-style JSONL)
# ============================================================================

def load_qa_data(json_path: str) -> Tuple[List[str], List[int]]:
    prompts, labels = [], []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if not (obj := line.strip()):
                continue
            obj = json.loads(obj)
            knowledge = obj.get("knowledge", "")
            question = obj.get("question", "")
            right_answer = obj.get("right_answer", "")
            hallucinated_answer = obj.get("hallucinated_answer", "")

            prompts.append(f"Knowledge: {knowledge}\nQuestion: {question}\nAnswer: {right_answer}")
            labels.append(0)
            prompts.append(f"Knowledge: {knowledge}\nQuestion: {question}\nAnswer: {hallucinated_answer}")
            labels.append(1)
    return prompts, labels


def load_dialogue_data(json_path: str) -> Tuple[List[str], List[int]]:
    prompts, labels = [], []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if not (obj := line.strip()):
                continue
            obj = json.loads(obj)
            knowledge = obj.get("knowledge", "")
            dialogue_history = obj.get("dialogue_history", "")
            right_response = obj.get("right_response", "")
            hallucinated_response = obj.get("hallucinated_response", "")

            prompts.append(f"Knowledge: {knowledge}\nDialogue: {dialogue_history}\nResponse: {right_response}")
            labels.append(0)
            prompts.append(f"Knowledge: {knowledge}\nDialogue: {dialogue_history}\nResponse: {hallucinated_response}")
            labels.append(1)
    return prompts, labels


def load_summarization_data(json_path: str) -> Tuple[List[str], List[int]]:
    prompts, labels = [], []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if not (obj := line.strip()):
                continue
            obj = json.loads(obj)
            document = obj.get("document", "")
            right_summary = obj.get("right_summary", "")
            hallucinated_summary = obj.get("hallucinated_summary", "")

            prompts.append(f"Document: {document}\nSummary: {right_summary}")
            labels.append(0)
            prompts.append(f"Document: {document}\nSummary: {hallucinated_summary}")
            labels.append(1)
    return prompts, labels


def load_general_data(json_path: str) -> Tuple[List[str], List[int]]:
    prompts, labels = [], []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if not (obj := line.strip()):
                continue
            obj = json.loads(obj)
            user_query = obj.get("user_query", "")
            chatgpt_response = obj.get("chatgpt_response", "")
            hallucination = obj.get("hallucination", "no")

            prompts.append(f"Query: {user_query}\nResponse: {chatgpt_response}")
            labels.append(1 if hallucination.lower() == "yes" else 0)
    return prompts, labels


def load_all_data(data_dir: str, dataset_type: str = "all") -> Tuple[List[str], List[int], Dict]:
    all_prompts, all_labels, data_stats = [], [], {}

    loaders = {
        "qa": ("qa_data.json", load_qa_data),
        "dialogue": ("dialogue_data.json", load_dialogue_data),
        "summarization": ("summarization_data.json", load_summarization_data),
        "general": ("general_data.json", load_general_data),
    }

    for name, (filename, loader) in loaders.items():
        if dataset_type in ["all", name]:
            path = os.path.join(data_dir, filename)
            if os.path.exists(path):
                prompts, labels = loader(path)
                all_prompts.extend(prompts)
                all_labels.extend(labels)
                data_stats[name] = {"correct": sum(1 for l in labels if l == 0),
                                     "hallucinated": sum(1 for l in labels if l == 1)}

    return all_prompts, all_labels, data_stats


def split_stratified(y: List[int], test_ratio: float, val_ratio: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    rng = random.Random(seed)
    idx0 = [i for i, yi in enumerate(y) if yi == 0]
    idx1 = [i for i, yi in enumerate(y) if yi == 1]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    def split(idx_list):
        n = len(idx_list)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        n_train = n - n_test - n_val
        return idx_list[:n_train], idx_list[n_train:n_train+n_val], idx_list[n_train+n_val:]

    tr0, va0, te0 = split(idx0)
    tr1, va1, te1 = split(idx1)

    train_idx = tr0 + tr1
    val_idx = va0 + va1
    test_idx = te0 + te1

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx, val_idx, test_idx


# ============================================================================
# ========== Training / CLI entrypoints
# ============================================================================

def train(args):
    """Full training + evaluation pipeline."""
    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Stage {args.stage} | device: {device}")
    print(f"Calibration method: {args.calibration_method}")

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, use_fast=True)

    # Load JSONL shards
    prompts, labels, data_stats = load_all_data(args.data_dir, args.dataset_type)
    print(f"Total samples: {len(prompts)}")
    for name, stats in data_stats.items():
        print(f"  {name}: non-hallucinated={stats['correct']}, hallucinated={stats['hallucinated']}")

    # Stratified split
    train_idx, val_idx, test_idx = split_stratified(labels, args.test_ratio, args.val_ratio, args.seed)

    train_prompts = [prompts[i] for i in train_idx]
    val_prompts = [prompts[i] for i in val_idx]
    test_prompts = [prompts[i] for i in test_idx]
    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]
    test_labels = [labels[i] for i in test_idx]

    # ===== Stage 2: uncertainty features =====
    if int(args.stage) >= 2:
        print("\nComputing uncertainty features...")
        train_unc = [compute_uncertainty_features(p) for p in train_prompts]
        val_unc = [compute_uncertainty_features(p) for p in val_prompts]
        test_unc = [compute_uncertainty_features(p) for p in test_prompts]

        train_unc_norm, _ = normalize_features(train_unc, train_unc)
        val_unc_norm, _ = normalize_features(train_unc, val_unc)
        test_unc_norm, _ = normalize_features(train_unc, test_unc)
    else:
        train_unc_norm = val_unc_norm = test_unc_norm = None

    # Dataset / loaders
    train_ds = HaluEvalDataset(train_prompts, train_labels, tokenizer, args.max_length, train_unc_norm)
    val_ds = HaluEvalDataset(val_prompts, val_labels, tokenizer, args.max_length, val_unc_norm)
    test_ds = HaluEvalDataset(test_prompts, test_labels, tokenizer, args.max_length, test_unc_norm)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_fn)

    # ===== Model =====
    encoder = AutoModel.from_pretrained(args.pretrained_model)
    hidden_size = encoder.config.hidden_size

    if int(args.stage) == 1:
        model = HallucinationPredictor(encoder, hidden_size, args.dropout, args.use_mean_pooling)
    else:
        model = EnhancedHallucinationPredictor(encoder, hidden_size, 6, args.dropout, args.use_mean_pooling)

    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False

    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                   lr=args.lr, weight_decay=args.weight_decay)

    # ===== Training loop =====
    best_auroc = -float('inf')
    best_path = os.path.join(args.output_dir, "best.pt")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            optimizer.zero_grad(set_to_none=True)

            kwargs = {
                'input_ids': batch.input_ids.to(device),
                'attention_mask': batch.attention_mask.to(device),
            }

            if int(args.stage) >= 2:
                kwargs.update({
                    'perplexity': batch.perplexity.to(device),
                    'token_entropy': batch.token_entropy.to(device),
                    'answer_length': batch.answer_length.to(device),
                    'answer_char_length': batch.answer_char_length.to(device),
                    'avg_confidence': batch.avg_confidence.to(device),
                    'sequence_entropy': batch.sequence_entropy.to(device),
                })

            logits = model(**kwargs)
            loss = loss_fn(logits, batch.label.to(device))
            loss.backward()

            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()

            epoch_loss += float(loss.detach().cpu().item())

        # Validation
        val_metrics = evaluate(model, val_loader, device, args.stage)
        print(f"Epoch {epoch+1}: Loss={epoch_loss/len(train_loader):.4f}, "
              f"Val AUROC={val_metrics['auroc']:.4f}, ECE={val_metrics['ece']:.4f}")

        # Checkpoint best AUROC
        if val_metrics['auroc'] > best_auroc:
            best_auroc = val_metrics['auroc']
            torch.save({
                "model_state_dict": model.state_dict(),
                "stage": args.stage,
                "pretrained_model": args.pretrained_model,
                "config": {
                    "max_length": args.max_length,
                    "dropout": args.dropout,
                    "use_mean_pooling": args.use_mean_pooling,
                }
            }, best_path)

        # CSV log
        log_path = os.path.join(args.output_dir, "train_log.csv")
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if epoch == 0:
                writer.writerow(["epoch", "loss", "val_auroc", "val_aupr", "val_ece"])
            writer.writerow([epoch+1, epoch_loss/len(train_loader),
                            val_metrics['auroc'], val_metrics['aupr'], val_metrics['ece']])

    # ===== Stage 3: calibration =====
    print(f"\n{'='*60}")
    print(f"Stage 3: fitting {args.calibration_method} calibration")
    print(f"{'='*60}")

    # Reload best weights
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])

    # Gather validation logits
    val_logits, val_labels_list = [], []
    model.eval()
    for batch in tqdm(val_loader, desc="Collect val logits"):
        kwargs = {'input_ids': batch.input_ids.to(device), 'attention_mask': batch.attention_mask.to(device)}
        if int(args.stage) >= 2:
            kwargs.update({
                'perplexity': batch.perplexity.to(device),
                'token_entropy': batch.token_entropy.to(device),
                'answer_length': batch.answer_length.to(device),
                'answer_char_length': batch.answer_char_length.to(device),
                'avg_confidence': batch.avg_confidence.to(device),
                'sequence_entropy': batch.sequence_entropy.to(device),
            })
        logits = model(**kwargs)
        val_logits.extend(logits.detach().cpu().numpy().tolist())
        val_labels_list.extend(batch.label.numpy().tolist())

    val_logits_np = np.array(val_logits)
    val_labels_np = np.array(val_labels_list)

    # Fit calibrator on validation predictions
    calibrator = None
    if args.calibration_method == "temperature":
        calibrator = TemperatureScaling()
        calibrator.to(device)
        logits_tensor = torch.tensor(val_logits_np, dtype=torch.float32, device=device)
        labels_tensor = torch.tensor(val_labels_np, dtype=torch.float32, device=device)
        optimizer = torch.optim.LBFGS(calibrator.parameters(), lr=0.01, max_iter=100)
        def closure():
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                calibrator(logits_tensor),
                labels_tensor)
            loss.backward()
            return loss
        optimizer.step(closure)
        torch.save(calibrator.state_dict(), os.path.join(args.output_dir, "calibrator.pt"))

    elif args.calibration_method == "platt":
        calibrator = PlattScaling()
        calibrator.to(device)
        logits_tensor = torch.tensor(val_logits_np, dtype=torch.float32, device=device)
        labels_tensor = torch.tensor(val_labels_np, dtype=torch.float32, device=device)
        optimizer = torch.optim.LBFGS(calibrator.parameters(), lr=0.01, max_iter=100)
        def closure():
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                calibrator(logits_tensor),
                labels_tensor)
            loss.backward()
            return loss
        optimizer.step(closure)
        torch.save(calibrator.state_dict(), os.path.join(args.output_dir, "calibrator.pt"))

    elif args.calibration_method == "isotonic":
        calibrator = IsotonicCalibrator()
        probs = 1 / (1 + np.exp(-val_logits_np))
        calibrator.fit(probs, val_labels_np)

    # ===== Test evaluation =====
    print("\nEvaluating on test split...")
    test_metrics = evaluate(model, test_loader, device, args.stage, calibrator)

    # Persist metrics JSON
    results = {
        "stage": args.stage,
        "calibration": args.calibration_method,
        "dataset": args.dataset_type,
        "data_stats": data_stats,
        "split": {"train": len(train_prompts), "val": len(val_prompts), "test": len(test_prompts)},
        "metrics": {k: v for k, v in test_metrics.items() if k not in ['raw_probs', 'raw_logits', 'labels']},
    }

    with open(os.path.join(args.output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Raw predictions for downstream comparisons
    predictions = {
        "probs": test_metrics['raw_probs'],
        "labels": test_metrics['labels'],
    }
    with open(os.path.join(args.output_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(predictions, f)

    # ===== Stage 3: plots =====
    if int(args.stage) >= 3:
        print("\nSaving diagnostic plots...")
        plot_cav_curve(model, test_loader, device,
                      os.path.join(args.output_dir, "cav_curve.png"),
                      int(args.stage), calibrator)
        plot_calibration_curve(np.array(test_metrics['raw_probs']),
                               np.array(test_metrics['labels']),
                               os.path.join(args.output_dir, "calibration_curve.png"))

    # ===== Stage 3: trust signal demo =====
    print(f"\n{'='*60}")
    print("Stage 3: Trust signal demo (up to 2 samples per HIGH / MEDIUM / LOW trust tier)")
    print(f"{'='*60}")

    raw_probs_list = test_metrics["raw_probs"]

    def _trust_tier(idx: int) -> str:
        return generate_trust_signal(1.0 - float(raw_probs_list[idx]))[0]

    demo_indices = select_demo_indices_by_trust_level(
        test_prompts,
        raw_probs_list,
        trust_fn=_trust_tier,
        n_per_level=2,
    )
    tier_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for i in demo_indices:
        tier_counts[_trust_tier(i)] += 1
    print(
        f"Trust-tier picks — HIGH: {tier_counts['HIGH']}, "
        f"MEDIUM: {tier_counts['MEDIUM']}, LOW: {tier_counts['LOW']} "
        "(tiers from confidence thresholds on calibrated outputs)"
    )

    for i in demo_indices:
        prompt = test_prompts[i]
        prob = raw_probs_list[i]

        answer = _extract_demo_answer(prompt)
        if answer is None:
            ans_start = prompt.lower().find("answer:")
            answer = prompt[ans_start + 7 :].strip() if ans_start != -1 else "[N/A]"

        print(format_answer_output(
            answer, 1 - prob, prob,
            is_calibrated=args.calibration_method != "none",
        ))

    # Persist tokenizer next to checkpoints
    tokenizer.save_pretrained(args.output_dir)

    print(f"\nDone. Artifacts directory: {args.output_dir}")
    print(f"Best Val AUROC: {best_auroc:.4f}")
    print(f"Test AUROC: {test_metrics['auroc']:.4f}")
    print(f"Test ECE: {test_metrics['ece']:.4f}")


def inference(args):
    """Load checkpoint and run a single-string demo."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Restore encoder + heads
    ckpt = torch.load(os.path.join(args.model_path, "best.pt"), map_location="cpu")
    stage = ckpt.get("stage", 1)
    config = ckpt.get("config", {})

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    encoder = AutoModel.from_pretrained(ckpt.get("pretrained_model", "distilbert-base-uncased"))
    hidden_size = encoder.config.hidden_size

    if int(stage) == 1:
        model = HallucinationPredictor(encoder, hidden_size, config.get("dropout", 0.1),
                                       config.get("use_mean_pooling", False))
    else:
        model = EnhancedHallucinationPredictor(encoder, hidden_size, 6,
                                               config.get("dropout", 0.1),
                                               config.get("use_mean_pooling", False))

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Optional calibrator weights
    calibrator = None
    cal_path = os.path.join(args.model_path, "calibrator.pt")
    if os.path.exists(cal_path) and args.use_calibration:
        if args.calibration_method == "temperature":
            calibrator = TemperatureScaling()
        elif args.calibration_method == "platt":
            calibrator = PlattScaling()
        calibrator.load_state_dict(torch.load(cal_path, map_location="cpu"))
        calibrator.to(device)

    # Run forward pass on CLI-provided text
    if args.input:
        enc = tokenizer(args.input, truncation=True, max_length=config.get("max_length", 256),
                       return_tensors="pt")
        batch = collate_fn([{
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "label": torch.tensor(0),
        }])

        if int(stage) >= 2:
            unc = compute_uncertainty_features(args.input)
            batch.perplexity = torch.tensor([[unc['perplexity']]])
            batch.token_entropy = torch.tensor([[unc['token_entropy']]])
            batch.answer_length = torch.tensor([[unc['answer_length']]])
            batch.answer_char_length = torch.tensor([[unc['answer_char_length']]])
            batch.avg_confidence = torch.tensor([[unc['avg_confidence']]])
            batch.sequence_entropy = torch.tensor([[unc['sequence_entropy']]])

        probs, _ = predict(model, batch, device, stage, calibrator)
        prob = probs.item()
        confidence = 1 - prob

        ans_start = args.input.lower().find("answer:")
        answer = args.input[ans_start+7:].strip() if ans_start != -1 else args.input

        print(format_answer_output(answer, confidence, prob,
                                   is_calibrated=calibrator is not None))

    else:
        print("Provide text via --input for inference mode.")


def main():
    parser = argparse.ArgumentParser(description="Failure-aware hallucination detector (stages 1–3).")

    # Mode
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "inference"])

    # Stage selector
    parser.add_argument("--stage", type=int, default=3, choices=[1, 2, 3],
                        help="1=text-only, 2=+uncertainty features, 3=+calibration")

    # Paths
    parser.add_argument("--data_dir", type=str, default="HaluEval-Data")
    parser.add_argument("--output_dir", type=str, default="outputs_failure_aware")
    parser.add_argument("--model_path", type=str, default="outputs_failure_aware",
                        help="Checkpoint directory for inference.")

    # Data
    parser.add_argument("--dataset_type", type=str, default="all",
                        choices=["all", "qa", "dialogue", "summarization", "general"])

    # Model
    parser.add_argument("--pretrained_model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--use_mean_pooling", action="store_true")
    parser.add_argument("--freeze_encoder", action="store_true")

    # Optimization / splits
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)

    # Stage 3
    parser.add_argument("--calibration_method", type=str, default="temperature",
                        choices=["temperature", "platt", "isotonic", "none"])

    # Inference-only flags
    parser.add_argument("--input", type=str, default=None, help="Prompt string for inference mode.")
    parser.add_argument("--use_calibration", action="store_true", default=True,
                        help="Apply calibration head during inference.")
    parser.add_argument("--no_calibration", dest="use_calibration", action="store_false",
                        help="Skip calibration head.")

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        inference(args)


if __name__ == "__main__":
    main()
