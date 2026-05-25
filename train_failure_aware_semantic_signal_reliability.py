"""
Failure-Aware hallucination detector — integrated stages 1+2+3
Built on a Transformer encoder (DistilBERT).

Features:
- Stage 1: Semantic analysis (SemanticAnalyzer: similarity, entity overlap, knowledge coverage)
- Stage 2: Uncertainty proxies (perplexity, entropy, self-consistency, answer length)
- Stage 3: Calibration (temperature / Platt) + trust signal + CAV plots

Usage:
    # Stage 1 — semantic analyzer
    python train_failure_aware_semantic.py --stage 1

    # Stage 2 — uncertainty features
    python train_failure_aware_semantic.py --stage 2

    # Stage 3 — full pipeline + calibration + plots
    python train_failure_aware_semantic.py --stage 3

    # Inference + trust signal
    python train_failure_aware_semantic.py --mode inference --input "Knowledge: ... Question: ... Answer: ..."
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
from typing import Any, Dict, List, Optional, Tuple

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

from train_failure_aware import select_demo_indices_by_trust_level


# ============================================================================
# ========== Utility Functions
# ============================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================================
# ========== Stage 1: Semantic Analyzer Components
# ============================================================================

class SemanticAnalyzer:
    """
    Semantic analysis component for Stage 1.

    Implements:
    - Semantic similarity computation (Sentence-BERT style)
    - Entity extraction and overlap detection
    - Knowledge coverage analysis
    - Embedding diversity metrics
    """

    def __init__(self, model_name: str = "distilbert-base-uncased", device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.to(device)
        self.encoder.eval()

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to embeddings using mean pooling."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.encoder(**inputs)
            hidden_states = outputs.last_hidden_state[0]  # [seq_len, hidden]

        # Mean pooling
        attention_mask = inputs['attention_mask'][0]
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=0)
        sum_mask = torch.clamp(mask_expanded.sum(dim=0), min=1e-9)
        embedding = (sum_embeddings / sum_mask).cpu().numpy()

        return embedding

    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = self.encode_text(text1)
        emb2 = self.encode_text(text2)

        # Normalize
        emb1 = emb1 / (np.linalg.norm(emb1) + 1e-9)
        emb2 = emb2 / (np.linalg.norm(emb2) + 1e-9)

        return float(np.dot(emb1, emb2))

    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities using simple pattern matching."""
        patterns = {
            'PERSON': r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',
            'LOCATION': r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:City|Country|State|River|Lake|Mountain|Ocean)\b)?)',
            'DATE': r'\b(\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})\b',
            'NUMBER': r'\b(\d+(?:\.\d+)?(?:\s*(?:%|million|billion|thousand|kg|lb|km|mi|meters|feet))?)\b',
            'ORGANIZATION': r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:University|Company|Institute|Organization|Corp|Inc)\b)?)',
        }

        entities = []
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            entities.extend(matches)

        return list(set(entities))

    def compute_entity_overlap(self, answer: str, knowledge: str) -> Dict[str, float]:
        """Compute entity-based features between answer and knowledge."""
        answer_entities = set(e.lower() for e in self.extract_entities(answer))
        knowledge_entities = set(e.lower() for e in self.extract_entities(knowledge))

        if len(answer_entities) == 0:
            return {
                'entity_overlap_ratio': 0.0,
                'entity_overlap_count': 0,
                'answer_entity_count': 0,
                'knowledge_entity_count': len(knowledge_entities),
                'entity_coverage': 0.0
            }

        overlap = answer_entities & knowledge_entities
        overlap_ratio = len(overlap) / len(answer_entities) if answer_entities else 0.0
        coverage = len(overlap) / len(knowledge_entities) if knowledge_entities else 0.0

        return {
            'entity_overlap_ratio': overlap_ratio,
            'entity_overlap_count': len(overlap),
            'answer_entity_count': len(answer_entities),
            'knowledge_entity_count': len(knowledge_entities),
            'entity_coverage': coverage
        }

    def compute_knowledge_coverage(self, answer: str, knowledge: str) -> Dict[str, float]:
        """Compute how well the answer covers the knowledge base."""
        answer_tokens = set(answer.lower().split())
        knowledge_tokens = set(w.lower().strip('.,!?;:') for w in knowledge.split())

        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall'}

        answer_keywords = answer_tokens - stopwords
        knowledge_keywords = knowledge_tokens - stopwords

        if len(answer_keywords) == 0:
            return {
                'knowledge_overlap': 0.0,
                'knowledge_coverage': 0.0,
                'answer_specificity': 0.0
            }

        overlap = answer_keywords & knowledge_keywords

        return {
            'knowledge_overlap': len(overlap) / len(answer_keywords) if answer_keywords else 0.0,
            'knowledge_coverage': len(overlap) / len(knowledge_keywords) if knowledge_keywords else 0.0,
            'answer_specificity': len(answer_keywords) / max(len(knowledge_keywords), 1)
        }


# ============================================================================
# ========== Stage 1: Dataset with Semantic Features
# ============================================================================

class HaluEvalDataset(Dataset):
    """Dataset shared by Stage 1 and Stage 2."""
    def __init__(self, prompts: List[str], labels: List[int], tokenizer, max_length: int,
                 semantic_features: Optional[List[Dict]] = None,
                 uncertainty_features: Optional[List[Dict]] = None):
        self.items = []
        for i, (prompt, label) in enumerate(zip(prompts, labels)):
            item = {"prompt": prompt, "label": label}
            if semantic_features is not None:
                item["semantic"] = semantic_features[i]
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
        if "semantic" in item:
            result["semantic"] = {k: torch.tensor(v, dtype=torch.float32) for k, v in item["semantic"].items()}
        if "uncertainty" in item:
            result["uncertainty"] = {k: torch.tensor(v, dtype=torch.float32) for k, v in item["uncertainty"].items()}
        return result


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    label: torch.Tensor
    # Stage 1: semantic tensors
    semantic_similarity: Optional[torch.Tensor] = None
    entity_overlap_ratio: Optional[torch.Tensor] = None
    knowledge_overlap: Optional[torch.Tensor] = None
    entity_coverage: Optional[torch.Tensor] = None
    # Stage 2: uncertainty tensors
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

    # Stage 1: semantic tensors
    if "semantic" in features[0]:
        batch.semantic_similarity = torch.tensor([f["semantic"].get("semantic_similarity", 0.5) for f in features], dtype=torch.float32)
        batch.entity_overlap_ratio = torch.tensor([f["semantic"].get("entity_overlap_ratio", 0.0) for f in features], dtype=torch.float32)
        batch.knowledge_overlap = torch.tensor([f["semantic"].get("knowledge_overlap", 0.5) for f in features], dtype=torch.float32)
        batch.entity_coverage = torch.tensor([f["semantic"].get("entity_coverage", 0.0) for f in features], dtype=torch.float32)

    # Stage 2: uncertainty tensors
    if "uncertainty" in features[0]:
        batch.perplexity = torch.stack([f["uncertainty"].get("perplexity", 0) for f in features])
        batch.token_entropy = torch.stack([f["uncertainty"].get("token_entropy", 0) for f in features])
        batch.answer_length = torch.stack([f["uncertainty"].get("answer_length", 0) for f in features])
        batch.answer_char_length = torch.stack([f["uncertainty"].get("answer_char_length", 0) for f in features])
        batch.avg_confidence = torch.stack([f["uncertainty"].get("avg_confidence", 0.5) for f in features])
        batch.sequence_entropy = torch.stack([f["uncertainty"].get("sequence_entropy", 0) for f in features])

    return batch


# ============================================================================
# ========== Stage 1: Semantic-Aware Model
# ============================================================================

class SemanticAwareHallucinationPredictor(nn.Module):
    """
    Stage 1 model — semantics-aware classifier

    Highlights:
    - Text embeddings fused with semantic statistics
    - Semantic stats: similarity, entity overlap, knowledge coverage
    """
    def __init__(self, encoder: AutoModel, hidden_size: int,
                 n_semantic_features: int = 4, dropout: float = 0.1,
                 use_mean_pooling: bool = False):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.use_mean_pooling = use_mean_pooling

        # Semantic projection
        self.semantic_proj = nn.Sequential(
            nn.Linear(n_semantic_features, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, hidden_size // 4),
        )

        # Fusion MLP
        fusion_dim = hidden_size + hidden_size // 4
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
        )

        self.head = nn.Linear(hidden_size // 2, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                semantic_similarity: torch.Tensor, entity_overlap_ratio: torch.Tensor,
                knowledge_overlap: torch.Tensor, entity_coverage: torch.Tensor) -> torch.Tensor:
        # Text embedding
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state

        if self.use_mean_pooling:
            mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
            text_emb = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            text_emb = last_hidden[:, 0, :]

        # Semantic branch
        semantic_feats = torch.stack([
            semantic_similarity,
            entity_overlap_ratio,
            knowledge_overlap,
            entity_coverage
        ], dim=1)

        semantic_emb = self.semantic_proj(semantic_feats)

        # Fuse branches
        combined = torch.cat([text_emb, semantic_emb], dim=1)
        fused = self.fusion_layer(combined)

        return self.head(self.dropout(fused)).squeeze(-1)


# ============================================================================
# ========== Stage 1: Semantic Features Computation
# ============================================================================

def parse_prompt_components(prompt: str) -> Tuple[str, str, str]:
    """Parse prompts into knowledge / question / answer spans."""
    knowledge = ""
    question = ""
    answer = ""

    # Try multiple HaluEval layouts
    knowledge_match = re.search(r'[Kk]nowledge:\s*(.+?)(?=\n[Qq]uestion:|\nAnswer:|$)', prompt, re.DOTALL)
    question_match = re.search(r'[Qq]uestion:\s*(.+?)(?=\nAnswer:|$)', prompt, re.DOTALL)
    answer_match = re.search(r'Answer:\s*(.+?)$', prompt, re.DOTALL)

    # Dialogue-style prompts
    if not knowledge_match:
        knowledge_match = re.search(r'[Kk]nowledge:\s*(.+?)(?=\n[Dd]ialogue:|\n[Qq]uestion:|$)', prompt, re.DOTALL)
    if not question_match:
        dialogue_match = re.search(r'[Dd]ialog(?:ue)?:\s*(.+?)(?=\n[Rr]esponse:|$)', prompt, re.DOTALL)
        if dialogue_match:
            question = dialogue_match.group(1).strip()
    if not answer_match:
        response_match = re.search(r'[Rr]esponse:\s*(.+?)$', prompt, re.DOTALL)
        if response_match:
            answer = response_match.group(1).strip()

    # Summarization prompts
    if not knowledge_match:
        doc_match = re.search(r'[Dd]ocument:\s*(.+?)(?=\n[Ss]ummary:|$)', prompt, re.DOTALL)
        if doc_match:
            knowledge = doc_match.group(1).strip()
    if not answer_match:
        summary_match = re.search(r'[Ss]ummary:\s*(.+?)$', prompt, re.DOTALL)
        if summary_match:
            answer = summary_match.group(1).strip()

    # General-domain prompts
    if not knowledge_match:
        query_match = re.search(r'[Qq]uery:\s*(.+?)(?=\n[Rr]esponse:|$)', prompt, re.DOTALL)
        if query_match:
            question = query_match.group(1).strip()

    # Prefer regex captures
    if knowledge_match:
        knowledge = knowledge_match.group(1).strip()
    if question_match:
        question = question_match.group(1).strip()
    if answer_match:
        answer = answer_match.group(1).strip()

    return knowledge, question, answer


def compute_semantic_features(prompt: str, semantic_analyzer: Optional[SemanticAnalyzer] = None) -> Dict[str, float]:
    """
    Compute semantic statistics (Stage 1).

    Powered by SemanticAnalyzer:
    - Semantic similarity (answer vs knowledge)
    - Entity overlap ratio
    - Knowledge overlap / coverage
    - Entity coverage
    """
    knowledge, question, answer = parse_prompt_components(prompt)

    features = {}

    if semantic_analyzer is not None:
        # Encoder-backed analyzer
        if answer and knowledge:
            features['semantic_similarity'] = semantic_analyzer.compute_semantic_similarity(answer, knowledge)
            entity_info = semantic_analyzer.compute_entity_overlap(answer, knowledge)
            features['entity_overlap_ratio'] = entity_info['entity_overlap_ratio']
            features['entity_coverage'] = entity_info['entity_coverage']
            coverage_info = semantic_analyzer.compute_knowledge_coverage(answer, knowledge)
            features['knowledge_overlap'] = coverage_info['knowledge_overlap']
        else:
            features['semantic_similarity'] = 0.5
            features['entity_overlap_ratio'] = 0.0
            features['entity_coverage'] = 0.0
            features['knowledge_overlap'] = 0.5
    else:
        # Rule-based fallback
        if answer and knowledge:
            # Lexical overlap proxy for similarity
            answer_words = set(answer.lower().split())
            knowledge_words = set(knowledge.lower().split())
            stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                        'would', 'could', 'should', 'may', 'might', 'must', 'shall'}
            answer_keywords = answer_words - stopwords
            knowledge_keywords = knowledge_words - stopwords

            if len(answer_keywords) > 0:
                overlap = answer_keywords & knowledge_keywords
                features['semantic_similarity'] = len(overlap) / len(answer_keywords)
                features['knowledge_overlap'] = len(overlap) / max(len(knowledge_keywords), 1)
            else:
                features['semantic_similarity'] = 0.5
                features['knowledge_overlap'] = 0.5

            # Lightweight entity overlap
            answer_entities = set(re.findall(r'\b[A-Z][a-z]+\b', answer))
            knowledge_entities = set(re.findall(r'\b[A-Z][a-z]+\b', knowledge))
            if len(answer_entities) > 0:
                overlap = answer_entities & knowledge_entities
                features['entity_overlap_ratio'] = len(overlap) / len(answer_entities)
                features['entity_coverage'] = len(overlap) / max(len(knowledge_entities), 1)
            else:
                features['entity_overlap_ratio'] = 0.0
                features['entity_coverage'] = 0.0
        else:
            features['semantic_similarity'] = 0.5
            features['entity_overlap_ratio'] = 0.0
            features['entity_coverage'] = 0.0
            features['knowledge_overlap'] = 0.5

    return features


def normalize_semantic_features(train_features: List[Dict], target_features: List[Dict]) -> Tuple[List[Dict], Dict]:
    """Normalize semantic features using training-set statistics."""
    keys = ['semantic_similarity', 'entity_overlap_ratio', 'knowledge_overlap', 'entity_coverage']

    stats = {}
    for key in keys:
        values = [f.get(key, 0.5) for f in train_features]
        stats[key] = {'mean': np.mean(values), 'std': np.std(values) + 1e-8}

    normalized = []
    for f in target_features:
        norm_f = {}
        for key in keys:
            val = f.get(key, 0.5)
            norm_f[key] = (val - stats[key]['mean']) / stats[key]['std']
        normalized.append(norm_f)

    return normalized, stats


# ============================================================================
# ========== Stage 2: Enhanced model + uncertainty features
# ============================================================================

class EnhancedHallucinationPredictor(nn.Module):
    """
    Stage 2 model — adds uncertainty fusion on top of Stage 1 semantics

    Inputs: text embedding + semantic statistics + uncertainty statistics.
    Output: P(hallucination).
    """
    def __init__(self, encoder: AutoModel, hidden_size: int,
                 n_semantic_features: int = 4,
                 n_uncertainty_features: int = 6, dropout: float = 0.1,
                 use_mean_pooling: bool = False):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.use_mean_pooling = use_mean_pooling

        # Stage 1 semantic projection
        self.semantic_proj = nn.Sequential(
            nn.Linear(n_semantic_features, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, hidden_size // 8),
        )

        # Stage 2 uncertainty projection
        self.uncertainty_proj = nn.Sequential(
            nn.Linear(n_uncertainty_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, hidden_size // 8),
        )

        # Fusion MLP
        fusion_dim = hidden_size + hidden_size // 8 + hidden_size // 8
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
        )

        self.head = nn.Linear(hidden_size // 2, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                semantic_similarity: torch.Tensor, entity_overlap_ratio: torch.Tensor,
                knowledge_overlap: torch.Tensor, entity_coverage: torch.Tensor,
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

        # Semantic branch
        semantic_feats = torch.stack([
            semantic_similarity,
            entity_overlap_ratio,
            knowledge_overlap,
            entity_coverage
        ], dim=1)
        semantic_emb = self.semantic_proj(semantic_feats)

        # Uncertainty branch
        uncertainty_feats = torch.stack([
            perplexity, token_entropy,
            answer_length / 1000.0, answer_char_length / 10000.0,
            avg_confidence, sequence_entropy / 10.0
        ], dim=1)
        uncertainty_emb = self.uncertainty_proj(uncertainty_feats)

        # Fuse branches
        combined = torch.cat([text_emb, semantic_emb, uncertainty_emb], dim=1)
        fused = self.fusion_layer(combined)

        return self.head(self.dropout(fused)).squeeze(-1)


# ============================================================================
# ========== Stage 2: Uncertainty feature computation
# ============================================================================

def compute_uncertainty_features(prompt: str) -> Dict[str, float]:
    """
    Heuristic uncertainty features for the answer span.

    Note: production stacks can swap in LLM log-prob APIs for real perplexity / entropy.
    Here we estimate from shallow statistics.
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

    # Perplexity proxy (coverage + length + light stochastic jitter)
    # Low coverage -> higher perplexity; longer answers -> slightly higher perplexity
    knowledge_cov = features.get('knowledge_overlap', 0.5)
    base_perplexity = 1.0 + (1.0 - knowledge_cov) * 8.0  # Lower coverage -> higher perplexity
    if features['answer_length'] > 10:
        base_perplexity *= 1.2  # Longer answers bump perplexity
    elif features['answer_length'] <= 3:
        base_perplexity *= 0.8  # Very short answers reduce perplexity
    # Inject jitter to mimic estimator noise
    features['perplexity'] = base_perplexity + random.uniform(-0.5, 0.5)
    
    # Token entropy proxy (lexical diversity + numeric density + sentence count)
    # Many digits -> higher uncertainty; many sentences -> slightly higher entropy
    num_density = features.get('numeric_density', 0)
    sent_count = features.get('sentence_count', 1)
    base_entropy = 0.5 + num_density * 2.0  # More digits -> higher entropy
    base_entropy += min(sent_count / 10.0, 1.0)  # More sentences slightly raise entropy
    if features['answer_length'] > 20:
        base_entropy *= 1.3  # Very long answers increase entropy
    elif features['answer_length'] <= 5:
        base_entropy *= 0.7  # Short answers shrink entropy
    # Add stochastic jitter
    features['token_entropy'] = base_entropy + random.uniform(-0.3, 0.3)

    # Confidence proxy
    features['avg_confidence'] = features['knowledge_overlap']
    features['sequence_entropy'] = features['token_entropy'] * features['answer_length']

    return features


def normalize_uncertainty_features(train_features: List[Dict], target_features: List[Dict]) -> Tuple[List[Dict], Dict]:
    """Normalize tensors using training-set moments."""
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
# ========== Stage 2.5: Signal reliability analyzer
# ============================================================================

class SignalReliabilityAnalyzer:
    """
    Aggregates heterogeneous detectors into a reliability score.
    
    Responsibilities:
    1. Score intra-signal agreement
    2. Surface contradictory cues
    3. Produce a composite reliability metric
    4. Nudge headline confidence
    
    Runs after Stage 2 features and before Stage 3 calibration.
    """
    
    def __init__(self, 
                 consistency_weight: float = 0.4,
                 conflict_penalty: float = 0.3,
                 confidence_agreement_weight: float = 0.3):
        """
        Args:
            consistency_weight: weight on agreement term
            conflict_penalty: penalty when cues clash
            confidence_agreement_weight: weight on confidence agreement
        """
        self.consistency_weight = consistency_weight
        self.conflict_penalty = conflict_penalty
        self.confidence_agreement_weight = confidence_agreement_weight
        
        # Quantization thresholds
        self.thresholds = {
            'semantic_similarity': {'high': 0.7, 'low': 0.3},
            'perplexity': {'high': 30, 'low': 10},
            'token_entropy': {'high': 2.0, 'low': 0.5},
            'knowledge_overlap': {'high': 0.6, 'low': 0.2},
            'entity_overlap': {'high': 0.5, 'low': 0.15},
        }
    
    def _classify_signal(self, signal_name: str, value: float) -> str:
        """Bucket raw telemetry into high / moderate / low bins."""
        if signal_name not in self.thresholds:
            return 'moderate'
        
        thresh = self.thresholds[signal_name]
        
        # Low perplexity / entropy generally imply stable generations
        if signal_name in ['perplexity', 'token_entropy']:
            if value <= thresh['low']:
                return 'high'
            elif value <= thresh['high']:
                return 'moderate'
            else:
                return 'low'
        else:
            # Other cues: larger values imply reliability
            if value >= thresh['high']:
                return 'high'
            elif value >= thresh['low']:
                return 'moderate'
            else:
                return 'low'
    
    def _compute_consistency_score(self, features: Dict[str, float]) -> float:
        """
        Compute consistency among quantized cues.
        
        High consistency means cues mostly agree (all safe or all risky).
        Low consistency means mixed verdicts.
        """
        signals = {}
        
        # Quantize each telemetry channel
        if 'semantic_similarity' in features:
            signals['semantic'] = self._classify_signal('semantic_similarity', features['semantic_similarity'])
        if 'perplexity' in features:
            signals['perplexity'] = self._classify_signal('perplexity', features['perplexity'])
        if 'token_entropy' in features:
            signals['entropy'] = self._classify_signal('token_entropy', features['token_entropy'])
        if 'knowledge_overlap' in features:
            signals['knowledge'] = self._classify_signal('knowledge_overlap', features['knowledge_overlap'])
        if 'entity_overlap_ratio' in features:
            signals['entity'] = self._classify_signal('entity_overlap', features['entity_overlap_ratio'])
        
        if len(signals) < 2:
            return 0.5  # Not enough cues
        
        # Aggregate histogram
        high_count = sum(1 for v in signals.values() if v == 'high')
        low_count = sum(1 for v in signals.values() if v == 'low')
        moderate_count = sum(1 for v in signals.values() if v == 'moderate')
        
        total = len(signals)
        
        # Consistency ~= dominant-bin mass
        max_category = max(high_count, low_count, moderate_count)
        consistency = max_category / total
        
        # Penalize when highs and lows co-exist
        if high_count > 0 and low_count > 0:
            consistency *= (1 - min(high_count, low_count) / total)
        
        return consistency
    
    def _detect_conflicts(self, features: Dict[str, float]) -> List[str]:
        """
        Enumerate contradictory cue pairs.
        
        Returns:
            List of conflict descriptions
        """
        conflicts = []
        
        # 1. Semantic similarity vs perplexity clash
        if 'semantic_similarity' in features and 'perplexity' in features:
            sem = self._classify_signal('semantic_similarity', features['semantic_similarity'])
            perp = self._classify_signal('perplexity', features['perplexity'])
            
            if sem == 'high' and perp == 'low':
                conflicts.append("High semantic similarity yet high perplexity")
            elif sem == 'low' and perp == 'high':
                conflicts.append("Low semantic similarity yet low perplexity")
        
        # 2. Knowledge overlap vs semantic similarity clash
        if 'knowledge_overlap' in features and 'semantic_similarity' in features:
            know = self._classify_signal('knowledge_overlap', features['knowledge_overlap'])
            sem = self._classify_signal('semantic_similarity', features['semantic_similarity'])
            
            if abs(features['knowledge_overlap'] - features['semantic_similarity']) > 0.4:
                conflicts.append("Large gap between knowledge overlap and semantic similarity")
        
        # 3. Entropy vs perplexity clash
        if 'token_entropy' in features and 'perplexity' in features:
            if features['perplexity'] > 30 and features['token_entropy'] < 0.5:
                conflicts.append("High perplexity but low entropy")
            elif features['perplexity'] < 10 and features['token_entropy'] > 2.0:
                conflicts.append("Low perplexity but high entropy")
        
        return conflicts
    
    def _compute_confidence_agreement(self, 
                                      features: Dict[str, float], 
                                      model_confidence: float) -> float:
        """
        Score agreement between softmax confidence and cues.
        
        Returns:
            agreement score: 0-1
            Larger values imply calibrated confidence.
        """
        # Majority vote among quantized cues
        high_count = 0
        low_count = 0
        
        if 'semantic_similarity' in features:
            if features['semantic_similarity'] > 0.7:
                high_count += 1
            elif features['semantic_similarity'] < 0.3:
                low_count += 1
                
        if 'perplexity' in features:
            if features['perplexity'] < 10:
                high_count += 1
            elif features['perplexity'] > 30:
                low_count += 1
                
        if 'knowledge_overlap' in features:
            if features['knowledge_overlap'] > 0.6:
                high_count += 1
            elif features['knowledge_overlap'] < 0.2:
                low_count += 1
        
        signals_reliable = high_count > low_count
        signals_unreliable = low_count > high_count
        
        # Confidence agreement
        if (signals_reliable and model_confidence > 0.7) or \
           (signals_unreliable and model_confidence < 0.3):
            return 1.0  # Strong agreement
        elif (signals_reliable and model_confidence > 0.5) or \
             (signals_unreliable and model_confidence < 0.5):
            return 0.7  # Mild agreement
        elif model_confidence > 0.4 and model_confidence < 0.6:
            return 0.5  # Model on the fence
        else:
            return 0.2  # Confidence contradicts cues
    
    def analyze(self, 
                features: Dict[str, float], 
                model_confidence: float) -> Dict[str, Any]:
        """
        Primary analysis entrypoint.
        
        Args:
            features: fused Stage 1 + Stage 2 tensors
            model_confidence: model confidence (1 - P(hallucination))
        
        Returns:
            {
                'reliability_score': float,       # composite reliability in [0,1]
                'trust_level': str,               # HIGH/MEDIUM/LOW
                'consistent_signals': List[str], # cues voting trustworthy
                'conflicting_signals': List[str], # cues voting risky
                'conflicts': List[str],           # natural-language summaries
                'confidence_adjustment': float,   # additive delta on confidence
                'adjusted_confidence': float,     # headline confidence after adjustment
                'reasoning': List[str],           # diagnostic breadcrumbs
            }
        """
        reasoning = []
        
        # 1. Consistency score
        consistency_score = self._compute_consistency_score(features)
        reasoning.append(f"Signal consistency: {consistency_score:.2f}")
        
        # 2. Conflict mining
        conflicts = self._detect_conflicts(features)
        reasoning.append(f"Detected {len(conflicts)} conflicts")
        
        # 3. Confidence agreement
        agreement = self._compute_confidence_agreement(features, model_confidence)
        reasoning.append(f"Confidence agreement: {agreement:.2f}")
        
        # 4. Blend into reliability_score
        reliability_score = (
            consistency_score * self.consistency_weight +
            (1 - len(conflicts) * 0.2) * self.conflict_penalty +
            agreement * self.confidence_agreement_weight
        )
        reliability_score = max(0.0, min(1.0, reliability_score))
        
        # 5. Map score -> trust tier
        if reliability_score >= 0.7 and len(conflicts) == 0:
            trust_level = "HIGH"
            reasoning.append("Verdict: cues agree; no conflicts")
        elif reliability_score >= 0.5:
            trust_level = "MEDIUM"
            reasoning.append("Verdict: moderately reliable cues")
        else:
            trust_level = "LOW"
            reasoning.append("Verdict: unreliable cues or unresolved conflicts")
        
        # 6. Confidence nudge
        if len(conflicts) > 0:
            confidence_adjustment = -0.1 * len(conflicts)
        elif reliability_score > 0.8:
            confidence_adjustment = 0.05
        else:
            confidence_adjustment = 0.0
        
        adjusted_confidence = model_confidence + confidence_adjustment
        adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
        
        # 7. Enumerate cue summaries
        consistent_signals = []
        conflicting_signals = []
        
        if 'semantic_similarity' in features:
            if features['semantic_similarity'] > 0.7:
                consistent_signals.append("semantic_similarity(high)")
            elif features['semantic_similarity'] < 0.3:
                conflicting_signals.append("semantic_similarity(low)")
                
        if 'perplexity' in features:
            if features['perplexity'] < 10:
                consistent_signals.append("perplexity(low)")
            elif features['perplexity'] > 30:
                conflicting_signals.append("perplexity(high)")
                
        if 'knowledge_overlap' in features:
            if features['knowledge_overlap'] > 0.6:
                consistent_signals.append("knowledge_overlap(high)")
            elif features['knowledge_overlap'] < 0.2:
                conflicting_signals.append("knowledge_overlap(low)")
        
        if 'token_entropy' in features:
            if features['token_entropy'] <= 0.5:
                consistent_signals.append("token_entropy(low)")
            elif features['token_entropy'] >= 2.0:
                conflicting_signals.append("token_entropy(high)")
        
        if 'entity_overlap_ratio' in features:
            if features['entity_overlap_ratio'] > 0.5:
                consistent_signals.append("entity_overlap(high)")
            elif features['entity_overlap_ratio'] < 0.15:
                conflicting_signals.append("entity_overlap(low)")
        
        return {
            'reliability_score': reliability_score,
            'trust_level': trust_level,
            'consistent_signals': consistent_signals,
            'conflicting_signals': conflicting_signals,
            'conflicts': conflicts,
            'confidence_adjustment': confidence_adjustment,
            'adjusted_confidence': adjusted_confidence,
            'reasoning': reasoning,
        }
    
    def get_trust_signal(self, 
                        features: Dict[str, float], 
                        model_confidence: float) -> Tuple[str, str]:
        """
        Legacy-compatible trust formatter.
        
        Returns:
            (trust_level, message)
        """
        result = self.analyze(features, model_confidence)
        
        level = result['trust_level']
        conf = result['adjusted_confidence']
        
        if level == "HIGH":
            msg = f"✓ High reliability ({result['reliability_score']:.0%}); cues agree; confidence {conf:.1%}"
            if result['consistent_signals']:
                msg += f"\n  Reliable cues: {', '.join(result['consistent_signals'])}"
        elif level == "MEDIUM":
            msg = f"⚠ Medium reliability ({result['reliability_score']:.0%}); mostly aligned cues; confidence {conf:.1%}"
            if result['conflicts']:
                msg += f"\n  Conflict: {result['conflicts'][0]}"
        else:
            msg = f"✗ Low reliability ({result['reliability_score']:.0%}); conflicting or unreliable cues"
            if result['conflicts']:
                msg += f"\n  Conflicts: {', '.join(result['conflicts'])}"
            msg += f"\n  Recommendation: abstain or manually verify"
        
        return level, msg


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
# ========== Stage 3: Trust signal (signal reliability extension)
# ============================================================================

def generate_trust_signal(confidence: float,
                         threshold_high: float = 0.85,
                         threshold_low: float = 0.35,
                         reliability_info: Optional[Dict] = None) -> Tuple[str, str]:
    """
    Build headline trust tier + rationale.
    
    Args:
        confidence: softmax confidence
        threshold_high: high-confidence cutoff
        threshold_low: low-confidence cutoff
        reliability_info: optional analyzer payload
    """
    # Prefer analyzer-aware messaging when present
    if reliability_info is not None:
        trust_level = reliability_info.get('trust_level', 'MEDIUM')
        reliability_score = reliability_info.get('reliability_score', 0.5)
        adjusted_conf = reliability_info.get('adjusted_confidence', confidence)
        
        if trust_level == "HIGH":
            return "HIGH", f"✓ High reliability ({reliability_score:.0%}); aligned cues; confidence {adjusted_conf:.1%}"
        elif trust_level == "MEDIUM":
            return "MEDIUM", f"⚠ Medium reliability ({reliability_score:.0%}); confidence {adjusted_conf:.1%}"
        else:
            return "LOW", f"✗ Low reliability ({reliability_score:.0%}); conflicting cues — proceed cautiously"
    
    # Fallback to probability-only tiers
    if confidence >= threshold_high:
        return "HIGH", f"✓ High confidence ({confidence:.1%}); model trusts the answer."
    elif confidence >= threshold_low:
        return "MEDIUM", f"⚠ Medium confidence ({confidence:.1%}); please verify when possible."
    else:
        return "LOW", f"✗ Low confidence ({confidence:.1%}); treat as unreliable."


def format_answer_output(answer: str, confidence: float,
                         p_hallucination: float,
                         is_calibrated: bool = False,
                         reliability_info: Optional[Dict] = None) -> str:
    """
    Pretty-print answers plus extended trust diagnostics.
    
    Args:
        reliability_info: optional analyzer payload
    """
    trust_level, trust_msg = generate_trust_signal(confidence, reliability_info=reliability_info)

    calibration_note = ""
    if is_calibrated:
        calibration_note = f" [post-calibration confidence]"

    output = f"""
{'='*60}
Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}
{'='*60}
P(hallucination) = {p_hallucination:.3f}
"""
    
    # Show reliability-adjusted headline numbers
    if reliability_info is not None and reliability_info.get('adjusted_confidence') is not None:
        adj = reliability_info['adjusted_confidence']
        output += f"""Final confidence = {adj:.1%} [signal reliability aware]
  └─ Model confidence = {confidence:.1%} (calibrated)
  └─ Reliability score = {reliability_info.get('reliability_score', 0):.1%}
"""
        output += f"Trust level: {trust_level}\n"
        output += f"{trust_msg}\n"
    else:
        output += f"""Final confidence = {confidence:.1%}
"""
        output += f"Trust level: {trust_level}\n"
        output += f"{trust_msg}\n"
    
    # Optional verbose cue listings
    if reliability_info is not None:
        output += f"""
--- Signal reliability details ---
"""
        if reliability_info.get('consistent_signals'):
            output += f"Supportive cues: {', '.join(reliability_info['consistent_signals'])}\n"
        if reliability_info.get('conflicts'):
            output += f"Conflict cues: {', '.join(reliability_info['conflicts'])}\n"
    
    output += f"{'='*60}\n"
    return output


def _extract_demo_answer(prompt: str) -> Optional[str]:
    """Demo answer extraction aligned with train_failure_aware.py."""
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


def build_trust_sample_record(
    i: int,
    test_prompts: List[str],
    test_probs_arr: np.ndarray,
    test_labels: List[Any],
    test_sem: List[Dict[str, Any]],
    test_unc: List[Dict[str, Any]],
    stage: int,
    calibrated: bool,
    reliability_analyzer: SignalReliabilityAnalyzer,
    prompt_max_chars: int,
    include_formatted_terminal: bool,
) -> Dict[str, Any]:
    """One test-row payload for trust_demo_samples.json (demo slice or full test export)."""
    prompt = test_prompts[i]
    prob = float(test_probs_arr[i])

    answer = _extract_demo_answer(prompt)
    if not answer:
        ans_start = prompt.lower().find("answer:")
        answer = prompt[ans_start + 7 :].strip() if ans_start != -1 else ""
    if not answer:
        answer = "[N/A]"

    confidence = 1.0 - prob

    if int(stage) >= 1 and test_sem and i < len(test_sem):
        sem_feat = test_sem[i]
        all_features = {
            "semantic_similarity": sem_feat.get("semantic_similarity", 0.5),
            "entity_overlap_ratio": sem_feat.get("entity_overlap_ratio", 0),
            "knowledge_overlap": sem_feat.get("knowledge_overlap", 0.5),
            "entity_coverage": sem_feat.get("entity_coverage", 0),
        }
    else:
        all_features = {}

    if int(stage) >= 2 and test_unc and i < len(test_unc):
        unc_feat = test_unc[i]
        all_features.update({
            "perplexity": unc_feat.get("perplexity", 5.0),
            "token_entropy": unc_feat.get("token_entropy", 1.0),
            "answer_length": unc_feat.get("answer_length", 10),
        })

    reliability_info = reliability_analyzer.analyze(all_features, confidence)

    excerpt = prompt[:prompt_max_chars]
    if len(prompt) > prompt_max_chars:
        excerpt += "\n... [truncated]"

    row: Dict[str, Any] = {
        "test_split_index": int(i),
        "gold_label": int(test_labels[i]) if i < len(test_labels) else None,
        "trust_tier": reliability_info.get("trust_level"),
        "p_hallucination": prob,
        "model_confidence_correct": confidence,
        "answer_demo_line": answer,
        "prompt_excerpt": excerpt,
        "calibrated": calibrated,
        "signal_reliability": reliability_info,
    }
    if include_formatted_terminal:
        row["formatted_terminal_output"] = format_answer_output(
            answer, confidence, prob,
            is_calibrated=calibrated,
            reliability_info=reliability_info,
        )
    if int(stage) >= 1 and test_sem and i < len(test_sem) and test_sem[i]:
        row["semantic_features"] = test_sem[i]
    if int(stage) >= 2 and test_unc and i < len(test_unc) and test_unc[i]:
        row["uncertainty_features"] = test_unc[i]
    contributions = compute_signal_contributions(all_features, prob)
    if contributions:
        row["signal_contributions"] = contributions
    return row


def select_demo_indices_by_p_hallucination_quantiles(
    test_prompts: List[str],
    raw_probs: List[float],
    n_samples: int = 5,
    min_answer_chars: int = 3,
) -> List[int]:
    """Quantile-stratified sampling over P(hallucination)."""
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
    """Aggregate metrics dictionary."""
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
    """Forward pass helper used during train/eval loops."""
    model.eval()

    kwargs = {
        'input_ids': batch.input_ids.to(device),
        'attention_mask': batch.attention_mask.to(device),
    }

    stage_int = int(stage)

    # Stage 1: semantic tensors
    if stage_int >= 1:
        kwargs.update({
            'semantic_similarity': batch.semantic_similarity.to(device),
            'entity_overlap_ratio': batch.entity_overlap_ratio.to(device),
            'knowledge_overlap': batch.knowledge_overlap.to(device),
            'entity_coverage': batch.entity_coverage.to(device),
        })

    # Stage 2: uncertainty tensors
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

    # Calibration hook
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
              stage: int = 1, calibrator=None, loss_fn=None) -> Dict:
    """Run metric computation on a DataLoader."""
    model.eval()
    all_probs, all_labels, all_logits = [], [], []
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        probs, logits = predict(model, batch, device, stage, calibrator)
        all_probs.extend(probs.cpu().numpy().tolist())
        all_logits.extend(logits.cpu().numpy().tolist())
        all_labels.extend(batch.label.numpy().tolist())

        # Loss + accuracy bookkeeping
        if loss_fn is not None:
            labels = batch.label.to(device)
            loss = loss_fn(logits, labels)
            total_loss += float(loss.detach().cpu().item())
            num_batches += 1

    metrics = compute_metrics(np.array(all_probs), np.array(all_labels))
    metrics['raw_probs'] = all_probs
    metrics['raw_logits'] = all_logits
    metrics['labels'] = all_labels

    # Attach scalar summaries
    if num_batches > 0:
        metrics['loss'] = total_loss / num_batches
    else:
        metrics['loss'] = 0.0

    # Accuracy @ 0.5 probability threshold
    probs_np = np.array(all_probs)
    labels_np = np.array(all_labels)
    preds = (probs_np >= 0.5).astype(int)
    metrics['accuracy'] = float((preds == labels_np).mean())

    return metrics


# ============================================================================
# ========== Stage 3: Visualization helpers
# ============================================================================

def plot_cav_curve(model: nn.Module, dataloader: DataLoader, device: torch.device,
                    output_path: str, stage: int = 1, calibrator=None):
    """Coverage vs accuracy curve."""
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
# ========== Stage 3: Extended diagnostics
# ============================================================================

def plot_roc_curve(probs: np.ndarray, labels: np.ndarray, output_path: str):
    """ROC curve plot."""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'Model (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
    ax.fill_between(fpr, tpr, alpha=0.2, color='blue')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to {output_path}")


def plot_precision_recall_curve(probs: np.ndarray, labels: np.ndarray, output_path: str):
    """Precision–recall curve plot."""
    from sklearn.metrics import precision_recall_curve, auc, average_precision_score
    
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)
    avg_precision = average_precision_score(labels, probs)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(recall, precision, 'b-', linewidth=2, 
            label=f'Model (AP = {avg_precision:.4f})')
    ax.fill_between(recall, precision, alpha=0.2, color='blue')
    
    baseline = labels.mean()
    ax.axhline(y=baseline, color='r', linestyle='--', alpha=0.7, 
               label=f'Random Classifier ({baseline:.4f})')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curve', fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Precision-recall curve saved to {output_path}")


def plot_signal_importance(feature_names: List[str], importance_scores: List[float], 
                           output_path: str):
    """Bar chart of cue importance scores."""
    # Sort by magnitude
    sorted_pairs = sorted(zip(feature_names, importance_scores), key=lambda x: x[1], reverse=True)
    sorted_names, sorted_scores = zip(*sorted_pairs)
    
    colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(sorted_names)))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(sorted_names)), sorted_scores, color=colors)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=11)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('Signal Importance', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.4f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Signal importance plot saved to {output_path}")


def plot_feature_distribution(features: Dict[str, List[float]], 
                               labels: np.ndarray,
                               feature_name: str,
                               output_path: str):
    """Violin/KDE comparison for one feature."""
    # Labels: 0 = faithful, 1 = hallucinated
    correct_mask = labels == 0
    hallucination_mask = labels == 1
    
    feat_correct = np.array(features[feature_name])[correct_mask]
    feat_hallucination = np.array(features[feature_name])[hallucination_mask]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if len(feat_correct) == 0 or len(feat_hallucination) == 0:
        print(f"Warning: feature {feature_name} lacks usable samples — skipping plot")
        return
    
    bins = np.linspace(min(feat_correct.min(), feat_hallucination.min()),
                       max(feat_correct.max(), feat_hallucination.max()), 30)
    
    ax.hist(feat_hallucination, bins=bins, alpha=0.6, label='Hallucination (label=1)', 
            color='red', density=True)
    ax.hist(feat_correct, bins=bins, alpha=0.6, label='Correct (label=0)', 
            color='green', density=True)
    
    ax.set_xlabel(feature_name, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{feature_name} Distribution: Correct vs Hallucination', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Feature distribution plot [{feature_name}] saved to {output_path}")


def plot_confusion_matrix(probs: np.ndarray, labels: np.ndarray, output_path: str):
    """Confusion-matrix heatmap."""
    from sklearn.metrics import confusion_matrix
    
    threshold = 0.5
    preds = (probs >= threshold).astype(int)
    
    cm = confusion_matrix(labels, preds)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, shrink=0.8)
    
    classes = ['Hallucination', 'Correct']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticklabels(classes, fontsize=12)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center", fontsize=16, fontweight='bold',
                    color="white" if cm[i, j] > thresh else "black")
    
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix heatmap saved to {output_path}")


def _json_numpy_default(o: Any) -> Any:
    """json.dump default= handler for numpy scalars and arrays."""
    if isinstance(o, (np.floating, np.float32, np.float64)):
        return float(o)
    if isinstance(o, (np.integer, np.int64, np.int32)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def export_prediction_failure_cases_json(
    output_path: str,
    test_prompts: List[str],
    probs: np.ndarray,
    labels: np.ndarray,
    test_sem: List[Dict[str, Any]],
    test_unc: List[Dict[str, Any]],
    stage: int,
    decision_threshold: float = 0.5,
    max_per_error_type: int = 25,
    prompt_max_chars: int = 2000,
) -> None:
    """
    Write interpretable failure cases (FP / FN vs gold labels) using the same
    threshold as plot_confusion_matrix. FP = gold correct (0) but predicted
    hallucination (1); FN = gold hallucination (1) but predicted correct (0).

    If max_per_error_type <= 0, export every FP/FN on the test split (no cap).
    """
    labels_arr = np.asarray(labels).astype(int).reshape(-1)
    probs_arr = np.asarray(probs, dtype=np.float64).reshape(-1)
    preds = (probs_arr >= decision_threshold).astype(int)

    fp_idx = np.where((labels_arr == 0) & (preds == 1))[0]
    fn_idx = np.where((labels_arr == 1) & (preds == 0))[0]

    fp_order = fp_idx[np.argsort(-probs_arr[fp_idx])]
    fn_order = fn_idx[np.argsort(probs_arr[fn_idx])]

    if max_per_error_type <= 0:
        fp_sel = fp_order
        fn_sel = fn_order
    else:
        fp_sel = fp_order[:max_per_error_type]
        fn_sel = fn_order[:max_per_error_type]

    analyzer = SignalReliabilityAnalyzer()

    def features_for_index(i: int) -> Dict[str, float]:
        all_features: Dict[str, float] = {}
        if int(stage) >= 1 and test_sem and i < len(test_sem):
            sem_feat = test_sem[i]
            all_features = {
                "semantic_similarity": float(sem_feat.get("semantic_similarity", 0.5)),
                "entity_overlap_ratio": float(sem_feat.get("entity_overlap_ratio", 0)),
                "knowledge_overlap": float(sem_feat.get("knowledge_overlap", 0.5)),
                "entity_coverage": float(sem_feat.get("entity_coverage", 0)),
            }
        if int(stage) >= 2 and test_unc and i < len(test_unc):
            unc_feat = test_unc[i]
            all_features.update({
                "perplexity": float(unc_feat.get("perplexity", 5.0)),
                "token_entropy": float(unc_feat.get("token_entropy", 1.0)),
                "answer_length": float(unc_feat.get("answer_length", 10)),
            })
        return all_features

    def one_case(i: int, kind: str) -> Dict[str, Any]:
        prob = float(probs_arr[i])
        feats = features_for_index(i)
        conf = 1.0 - prob
        rel = analyzer.analyze(feats, conf) if feats else {}
        prompt = test_prompts[i] if i < len(test_prompts) else ""
        excerpt = prompt[:prompt_max_chars]
        if len(prompt) > prompt_max_chars:
            excerpt += "\n... [truncated]"
        row: Dict[str, Any] = {
            "test_split_index": int(i),
            "error_type": kind,
            "gold_label": int(labels_arr[i]),
            "predicted_label": int(preds[i]),
            "p_hallucination": prob,
            "model_confidence_correct": conf,
            "prompt_excerpt": excerpt,
        }
        if test_sem and i < len(test_sem) and test_sem[i]:
            row["semantic_features"] = test_sem[i]
        if test_unc and i < len(test_unc) and test_unc[i]:
            row["uncertainty_features"] = test_unc[i]
        if rel:
            row["signal_reliability"] = rel

        # --- Error Taxonomy ---
        sem_sim = float((test_sem[i] if test_sem and i < len(test_sem) else {}).get("semantic_similarity", 0.5))
        know_ov = float((test_sem[i] if test_sem and i < len(test_sem) else {}).get("knowledge_overlap", 0.5))
        conflicts_list = rel.get("conflicts", []) if rel else []
        if conflicts_list:
            tax_type   = "Type3_conflicting_signals"
            tax_reason = "Conflicting signals prevent reliable prediction: " + "; ".join(conflicts_list)
        elif know_ov < 0.2:
            tax_type   = "Type2_missing_info"
            tax_reason = f"Low knowledge coverage (knowledge_overlap={know_ov:.2f}) — hallucination via missing info"
        elif sem_sim >= 0.6 and kind == "false_negative":
            tax_type   = "Type1_semantic_failure"
            tax_reason = f"High semantic similarity (semantic_similarity={sem_sim:.2f}) but wrong fact — surface matches, content wrong"
        else:
            tax_type   = "Type2_missing_info"
            tax_reason = f"Moderate/low signals (semantic_similarity={sem_sim:.2f}, knowledge_overlap={know_ov:.2f})"
        row["error_taxonomy"] = {"type": tax_type, "reason": tax_reason}

        # --- Signal contributions ---
        contributions = compute_signal_contributions(feats, prob)
        if contributions:
            row["signal_contributions"] = contributions

        return row

    fps = [one_case(int(i), "false_positive") for i in fp_sel]
    fns = [one_case(int(i), "false_negative") for i in fn_sel]

    payload = {
        "meta": {
            "decision_threshold": decision_threshold,
            "labels": {"0": "correct / non-hallucination", "1": "hallucination"},
            "false_positive": "gold 0, predicted 1 (model false alarm)",
            "false_negative": "gold 1, predicted 0 (missed hallucination)",
            "max_per_error_type": max_per_error_type,
            "export_all_errors": max_per_error_type <= 0,
            "prompt_max_chars": prompt_max_chars,
            "counts_on_test_set": {
                "false_positive_total": int(len(fp_idx)),
                "false_negative_total": int(len(fn_idx)),
            },
        },
        "false_positives": fps,
        "false_negatives": fns,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_numpy_default)
    print(f"Prediction failure cases (interpretability) saved to {output_path}")


def export_trust_demo_samples_json(
    output_path: str,
    meta: Dict[str, Any],
    samples: List[Dict[str, Any]],
) -> None:
    """Persist trust-tier qualitative demos (same content as terminal Stage 3 block)."""
    payload = {"meta": meta, "samples": samples}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_numpy_default)
    print(f"Trust demo samples saved to {output_path}")


def plot_learning_curve(train_losses: List[float], val_losses: List[float],
                        train_accs: List[float], val_accs: List[float],
                        output_path: str):
    """Train vs validation learning curves."""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss subplot
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss', marker='o', markersize=4)
    ax1.plot(epochs, val_losses, 'r-', linewidth=2, label='Val Loss', marker='s', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy subplot
    ax2.plot(epochs, train_accs, 'b-', linewidth=2, label='Train Acc', marker='o', markersize=4)
    ax2.plot(epochs, val_accs, 'r-', linewidth=2, label='Val Acc', marker='s', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training & Validation Accuracy', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Learning curves saved to {output_path}")


def plot_signal_correlation_heatmap(signal_data: Dict[str, List[float]], 
                                    output_path: str):
    """Pairwise cue correlation heatmap."""
    signal_names = list(signal_data.keys())
    n_signals = len(signal_names)
    corr_matrix = np.zeros((n_signals, n_signals))
    
    # Pearson correlation matrix
    for i in range(n_signals):
        for j in range(n_signals):
            x = np.array(signal_data[signal_names[i]])
            y = np.array(signal_data[signal_names[j]])
            corr_matrix[i, j] = np.corrcoef(x, y)[0, 1]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix, interpolation='nearest', cmap='RdYlBu', vmin=-1, vmax=1)
    
    ax.set_xticks(range(n_signals))
    ax.set_yticks(range(n_signals))
    ax.set_xticklabels(signal_names, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(signal_names, fontsize=11)
    
    plt.colorbar(im, ax=ax, shrink=0.8, label='Correlation')
    
    for i in range(n_signals):
        for j in range(n_signals):
            text_color = "white" if abs(corr_matrix[i, j]) > 0.5 else "black"
            ax.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center',
                    fontsize=10, color=text_color)
    
    ax.set_title('Signal Correlation Heatmap', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Signal correlation heatmap saved to {output_path}")


def compute_signal_contributions(features: Dict[str, float],
                                  final_score: float) -> Dict[str, float]:
    """
    Approximate per-signal additive contribution to the final hallucination score.

    Positive value → signal pushes p_hallucination UP.
    Negative value → signal pushes p_hallucination DOWN.
    """
    direction: Dict[str, float] = {
        "semantic_similarity":  -1.0,
        "knowledge_overlap":    -1.0,
        "entity_overlap_ratio": -1.0,
        "entity_coverage":      -1.0,
        "perplexity":           +1.0,
        "token_entropy":        +1.0,
        "sequence_entropy":     +1.0,
    }
    norm_ranges: Dict[str, tuple] = {
        "semantic_similarity":  (0.0, 1.0),
        "knowledge_overlap":    (0.0, 1.0),
        "entity_overlap_ratio": (0.0, 1.0),
        "entity_coverage":      (0.0, 1.0),
        "perplexity":           (0.0, 50.0),
        "token_entropy":        (0.0, 4.0),
        "sequence_entropy":     (0.0, 10.0),
    }
    present = {k: v for k, v in features.items() if k in direction}
    if not present:
        return {}
    raw: Dict[str, float] = {}
    for name, val in present.items():
        lo, hi = norm_ranges.get(name, (0.0, 1.0))
        normed = max(0.0, min(1.0, (float(val) - lo) / max(hi - lo, 1e-9)))
        raw[name] = direction[name] * (normed - 0.5)
    total_signed = sum(raw.values())
    scale = (final_score - 0.5) / total_signed if abs(total_signed) > 1e-9 else 1.0
    return {k: round(float(v * scale), 4) for k, v in raw.items()}


def plot_signal_agreement_map(output_dir: str) -> None:
    """
    3×3 heatmap: semantic_similarity level × perplexity level → Trust Level.
    Complements signal_correlation.png (feature redundancy) with decision-logic.
    Output: signal_agreement_map.png
    """
    import matplotlib.patches as mpatches

    row_labels = ["High sim\n(≥0.7)", "Mid sim\n(0.3–0.7)", "Low sim\n(<0.3)"]
    col_labels  = ["Low perp (<10)\n[fluent]", "Mid perp (10–30)", "High perp (>30)\n[uncertain]"]

    grid = [
        [("HIGH",   "Cues agree:\nfluent + grounded"),
         ("MEDIUM", "Sim ok,\nsome uncertainty"),
         ("LOW",    "Conflict:\nhigh sim + high perp")],
        [("HIGH",   "Moderate:\nlow uncertainty"),
         ("MEDIUM", "Ambiguous:\nboth moderate"),
         ("MEDIUM", "Leaning low:\nmod sim + high perp")],
        [("LOW",    "Conflict:\nlow sim + low perp"),
         ("LOW",    "Weak:\nboth negative"),
         ("HIGH",   "Cues agree:\nboth unreliable")],
    ]
    color_map = {"HIGH": "#2ecc71", "MEDIUM": "#f39c12", "LOW": "#e74c3c"}

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 3); ax.set_ylim(0, 3)
    ax.set_xticks([0.5, 1.5, 2.5]); ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks([0.5, 1.5, 2.5]); ax.set_yticklabels(row_labels[::-1], fontsize=9)
    ax.set_xlabel("Perplexity Level", fontsize=11, labelpad=8)
    ax.set_ylabel("Semantic Similarity Level", fontsize=11, labelpad=8)
    ax.set_title("Signal Agreement Map\n(semantic_similarity × perplexity → Trust Level)",
                 fontsize=12, fontweight="bold", pad=12)

    for r in range(3):
        for c in range(3):
            trust, note = grid[r][c]
            rect = mpatches.FancyBboxPatch(
                (c + 0.05, 2 - r + 0.05), 0.90, 0.90,
                boxstyle="round,pad=0.03", linewidth=1.5,
                edgecolor="white", facecolor=color_map[trust], alpha=0.88,
            )
            ax.add_patch(rect)
            ax.text(c + 0.5, 2 - r + 0.63, trust,
                    ha="center", va="center", fontsize=11, fontweight="bold", color="white")
            ax.text(c + 0.5, 2 - r + 0.29, note,
                    ha="center", va="center", fontsize=7.8, color="white", style="italic")

    legend_handles = [
        mpatches.Patch(color=color_map["HIGH"],   label="HIGH trust"),
        mpatches.Patch(color=color_map["MEDIUM"], label="MEDIUM trust"),
        mpatches.Patch(color=color_map["LOW"],    label="LOW trust"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(1.22, 1.0), fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "signal_agreement_map.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Signal agreement map saved to {out_path}")


def plot_all_feature_distributions(features: Dict[str, List[float]], 
                                   labels: np.ndarray,
                                   output_dir: str):
    """Batch-generate distribution diagnostics."""
    for feature_name in features.keys():
        output_path = os.path.join(output_dir, f"feature_dist_{feature_name}.png")
        plot_feature_distribution(features, labels, feature_name, output_path)


# ============================================================================
# ========== Data loading
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


def load_truthfulqa_data(csv_path: str) -> Tuple[List[str], List[int]]:
    """
    Load TruthfulQA.csv and convert to (prompt, label) pairs.

    Non-hallucination (label=0): Best Answer + each Correct Answer
    Hallucination   (label=1):   each Incorrect Answer

    Prompt format matches load_qa_data:
      Knowledge: <Source>\\nQuestion: <Q>\\nAnswer: <A>
    """
    prompts: List[str] = []
    labels:  List[int] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            knowledge = (row.get("Source")   or "").strip()
            question  = (row.get("Question") or "").strip()
            if not question:
                continue
            prefix = f"Knowledge: {knowledge}\nQuestion: {question}\nAnswer: "
            best = (row.get("Best Answer") or "").strip()
            if best:
                prompts.append(prefix + best)
                labels.append(0)
            for ans in (row.get("Correct Answers") or "").split(";"):
                ans = ans.strip()
                if ans and ans != best:
                    prompts.append(prefix + ans)
                    labels.append(0)
            for ans in (row.get("Incorrect Answers") or "").split(";"):
                ans = ans.strip()
                if ans:
                    prompts.append(prefix + ans)
                    labels.append(1)
    return prompts, labels


def load_all_data(data_dir: str, dataset_type: str = "all",
                  exclude_task: str = "") -> Tuple[List[str], List[int], Dict]:
    """
    Load HaluEval sub-tasks.

    Args:
        dataset_type:  "all" | "qa" | "dialogue" | "summarization" | "general"
        exclude_task:  sub-task name to skip when dataset_type == "all"
                       e.g. exclude_task="summarization" → train on qa+dialogue+general
    """
    all_prompts, all_labels, data_stats = [], [], {}

    loaders = {
        "qa": ("qa_data.json", load_qa_data),
        "dialogue": ("dialogue_data.json", load_dialogue_data),
        "summarization": ("summarization_data.json", load_summarization_data),
        "general": ("general_data.json", load_general_data),
        "truthfulqa": ("TruthfulQA.csv", load_truthfulqa_data),
    }

    for name, (filename, loader) in loaders.items():
        if dataset_type not in ["all", name]:
            continue
        if exclude_task and name == exclude_task:
            continue
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
# ========== CLI
# ============================================================================

def train(args):
    """End-to-end training + evaluation driver."""
    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Stage {args.stage} | device: {device}")
    print(f"Calibration method: {args.calibration_method}")

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, use_fast=True)

    # Load prompts
    exclude = getattr(args, "exclude_task", "") or ""
    if exclude:
        print(f"Excluding sub-task from training: {exclude}")
    prompts, labels, data_stats = load_all_data(args.data_dir, args.dataset_type, exclude)
    
    # Optionally subsample each shard
    if args.max_samples is not None:
        max_per_type = args.max_samples // 4  # split equally across four splits
        type_indices = {}
        for i, p in enumerate(prompts):
            for dtype in ['qa', 'dialogue', 'summarization', 'general']:
                if dtype in p.lower()[:50]:
                    if dtype not in type_indices:
                        type_indices[dtype] = []
                    type_indices[dtype].append(i)
                    break
            else:
                if 'general' not in type_indices:
                    type_indices['general'] = []
                type_indices['general'].append(i)
        
        # Reservoir-style sampling
        sampled_indices = []
        for dtype, indices in type_indices.items():
            if len(indices) > max_per_type:
                import random
                random.seed(args.seed)
                sampled_indices.extend(random.sample(indices, max_per_type))
            else:
                sampled_indices.extend(indices)
        
        prompts = [prompts[i] for i in sampled_indices]
        labels = [labels[i] for i in sampled_indices]
        print(f"Samples after subsampling: {len(prompts)}")

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

    test_sem: List[Dict[str, Any]] = []
    test_unc: List[Dict[str, Any]] = []

    # ===== Stage 1: semantic features =====
    if int(args.stage) >= 1:
        print("\nComputing semantic features (Stage 1)...")
        # Semantic analyzer singleton
        semantic_analyzer = SemanticAnalyzer(args.pretrained_model, device)
        
        # tqdm-tracked loops
        from tqdm import tqdm
        
        print("  Train split...")
        train_sem = []
        for p in tqdm(train_prompts, desc="  Train"):
            train_sem.append(compute_semantic_features(p, semantic_analyzer))
        
        print("  Validation split...")
        val_sem = []
        for p in tqdm(val_prompts, desc="  Val"):
            val_sem.append(compute_semantic_features(p, semantic_analyzer))
        
        print("  Test split...")
        test_sem = []
        for p in tqdm(test_prompts, desc="  Test"):
            test_sem.append(compute_semantic_features(p, semantic_analyzer))

        train_sem_norm, _ = normalize_semantic_features(train_sem, train_sem)
        val_sem_norm, _ = normalize_semantic_features(train_sem, val_sem)
        test_sem_norm, _ = normalize_semantic_features(train_sem, test_sem)
    else:
        semantic_analyzer = None
        train_sem_norm = val_sem_norm = test_sem_norm = None

    # ===== Stage 2: uncertainty features =====
    if int(args.stage) >= 2:
        print("\nComputing uncertainty features (Stage 2)...")
        
        print("  Train split...")
        train_unc = []
        for p in tqdm(train_prompts, desc="  Train"):
            train_unc.append(compute_uncertainty_features(p))
        
        print("  Validation split...")
        val_unc = []
        for p in tqdm(val_prompts, desc="  Val"):
            val_unc.append(compute_uncertainty_features(p))
        
        print("  Test split...")
        test_unc = []
        for p in tqdm(test_prompts, desc="  Test"):
            test_unc.append(compute_uncertainty_features(p))

        train_unc_norm, _ = normalize_uncertainty_features(train_unc, train_unc)
        val_unc_norm, _ = normalize_uncertainty_features(train_unc, val_unc)
        test_unc_norm, _ = normalize_uncertainty_features(train_unc, test_unc)
    else:
        train_unc_norm = val_unc_norm = test_unc_norm = None

    # Dataset objects
    train_ds = HaluEvalDataset(train_prompts, train_labels, tokenizer, args.max_length,
                               train_sem_norm, train_unc_norm)
    val_ds = HaluEvalDataset(val_prompts, val_labels, tokenizer, args.max_length,
                             val_sem_norm, val_unc_norm)
    test_ds = HaluEvalDataset(test_prompts, test_labels, tokenizer, args.max_length,
                              test_sem_norm, test_unc_norm)

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
        # Stage 1 semantics-aware classifier
        model = SemanticAwareHallucinationPredictor(encoder, hidden_size, 4, args.dropout, args.use_mean_pooling)
    else:
        # Stage 2+: fusion model with semantics + uncertainty
        model = EnhancedHallucinationPredictor(encoder, hidden_size, 4, 6, args.dropout, args.use_mean_pooling)

    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False

    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                   lr=args.lr, weight_decay=args.weight_decay)

    # ===== Metric history =====
    train_history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
    }

    # ===== Optimization loop =====
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

            # Stage 1 tensors
            if int(args.stage) >= 1:
                kwargs.update({
                    'semantic_similarity': batch.semantic_similarity.to(device),
                    'entity_overlap_ratio': batch.entity_overlap_ratio.to(device),
                    'knowledge_overlap': batch.knowledge_overlap.to(device),
                    'entity_coverage': batch.entity_coverage.to(device),
                })

            # Stage 2 tensors
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

        # Validation pass
        val_metrics = evaluate(model, val_loader, device, args.stage, loss_fn=loss_fn)
        print(f"Epoch {epoch+1}: Loss={epoch_loss/len(train_loader):.4f}, "
              f"Val AUROC={val_metrics['auroc']:.4f}, ECE={val_metrics['ece']:.4f}")

        # Cheap train-set diagnostics
        train_metrics = evaluate(model, train_loader, device, args.stage, loss_fn=loss_fn)
        
        # Append CSV history
        train_history['train_loss'].append(epoch_loss / len(train_loader))
        train_history['val_loss'].append(val_metrics.get('loss', 0))
        train_history['train_acc'].append(train_metrics.get('accuracy', 0))
        train_history['val_acc'].append(val_metrics.get('accuracy', 0))

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
                },
                "train_history": train_history,
            }, best_path)

        # Persist CSV rows
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
        kwargs = {
            'input_ids': batch.input_ids.to(device),
            'attention_mask': batch.attention_mask.to(device)
        }

        # Stage 1 tensors
        if int(args.stage) >= 1:
            kwargs.update({
                'semantic_similarity': batch.semantic_similarity.to(device),
                'entity_overlap_ratio': batch.entity_overlap_ratio.to(device),
                'knowledge_overlap': batch.knowledge_overlap.to(device),
                'entity_coverage': batch.entity_coverage.to(device),
            })

        # Stage 2 uncertainty branch
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

    # Fit calibrator
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
    print("\nEvaluating test split...")
    test_metrics = evaluate(model, test_loader, device, args.stage, calibrator)

    # Serialize metrics JSON
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

    # Serialize raw probs / labels
    predictions = {
        "probs": test_metrics['raw_probs'],
        "labels": test_metrics['labels'],
    }
    with open(os.path.join(args.output_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(predictions, f)

    # ===== Stage 3: diagnostics =====
    if int(args.stage) >= 3:
        print("\nSaving plots...")
        
        # Cache tensors for plotting
        test_probs = np.array(test_metrics['raw_probs'])
        test_labels_arr = np.array(test_metrics['labels'])
        
        # 1. CAV
        plot_cav_curve(model, test_loader, device,
                      os.path.join(args.output_dir, "cav_curve.png"),
                      int(args.stage), calibrator)
        
        # 2. Calibration
        plot_calibration_curve(test_probs, test_labels_arr,
                               os.path.join(args.output_dir, "calibration_curve.png"))
        
        # 3. ROC
        plot_roc_curve(test_probs, test_labels_arr,
                       os.path.join(args.output_dir, "roc_curve.png"))
        
        # 4. Precision–recall
        plot_precision_recall_curve(test_probs, test_labels_arr,
                                    os.path.join(args.output_dir, "pr_curve.png"))
        
        # 5. Confusion matrix
        plot_confusion_matrix(test_probs, test_labels_arr,
                              os.path.join(args.output_dir, "confusion_matrix.png"))

        export_prediction_failure_cases_json(
            os.path.join(args.output_dir, "failure_cases.json"),
            test_prompts,
            test_probs,
            test_labels_arr,
            test_sem,
            test_unc,
            int(args.stage),
            max_per_error_type=int(args.max_failure_cases_per_type),
        )
        
        # 6. Feature tensors for histograms
        if test_sem and test_unc:
            # Concatenate per-split caches
            all_features_dict = {}
            for feature_key in ['semantic_similarity', 'perplexity', 'knowledge_overlap', 
                               'token_entropy', 'entity_overlap_ratio']:
                feature_values = []
                for i in range(len(test_labels_arr)):
                    if feature_key in ['semantic_similarity', 'entity_overlap_ratio', 
                                       'knowledge_overlap']:
                        if i < len(test_sem) and test_sem[i]:
                            feature_values.append(test_sem[i].get(feature_key, 0.5))
                        else:
                            feature_values.append(0.5)
                    else:
                        if i < len(test_unc) and test_unc[i]:
                            feature_values.append(test_unc[i].get(feature_key, 1.0))
                        else:
                            feature_values.append(1.0)
                all_features_dict[feature_key] = feature_values
            
            # 6. Batch histograms
            plot_all_feature_distributions(all_features_dict, test_labels_arr,
                                          args.output_dir)
            
            # 7. Correlation heatmap
            plot_signal_correlation_heatmap(all_features_dict,
                                           os.path.join(args.output_dir, "signal_correlation.png"))

            # 7b. Signal agreement map (decision-logic visualisation)
            plot_signal_agreement_map(args.output_dir)
            
            # 8. Importance derived from cue-label correlation
            signal_names = list(all_features_dict.keys())
            importance_scores = []
            for name in signal_names:
                corr = abs(np.corrcoef(all_features_dict[name], test_labels_arr)[0, 1])
                importance_scores.append(corr if not np.isnan(corr) else 0.0)
            plot_signal_importance(signal_names, importance_scores,
                                  os.path.join(args.output_dir, "signal_importance.png"))
        
        # 9. Learning curves when history dict exists
        train_history = ckpt.get("train_history", None)
        if train_history:
            plot_learning_curve(
                train_history.get('train_loss', []),
                train_history.get('val_loss', []),
                train_history.get('train_acc', []),
                train_history.get('val_acc', []),
                os.path.join(args.output_dir, "learning_curve.png")
            )

    # ===== Stage 3: trust demo =====
    print(f"\n{'='*60}")
    print("Stage 3: Trust demo (up to 2 samples per HIGH / MEDIUM / LOW trust tier)")
    print(f"{'='*60}")
    
    # Analyzer singleton
    reliability_analyzer = SignalReliabilityAnalyzer()
    
    test_probs_arr = np.array(test_metrics["raw_probs"])

    def _trust_tier_for_demo(i: int) -> str:
        prob = float(test_probs_arr[i])
        confidence = 1.0 - prob
        all_features: Dict[str, float] = {}
        if int(args.stage) >= 1 and test_sem and i < len(test_sem):
            sem_feat = test_sem[i]
            all_features = {
                "semantic_similarity": sem_feat.get("semantic_similarity", 0.5),
                "entity_overlap_ratio": sem_feat.get("entity_overlap_ratio", 0),
                "knowledge_overlap": sem_feat.get("knowledge_overlap", 0.5),
                "entity_coverage": sem_feat.get("entity_coverage", 0),
            }
        if int(args.stage) >= 2 and test_unc and i < len(test_unc):
            unc_feat = test_unc[i]
            all_features.update({
                "perplexity": unc_feat.get("perplexity", 5.0),
                "token_entropy": unc_feat.get("token_entropy", 1.0),
                "answer_length": unc_feat.get("answer_length", 10),
            })
        info = reliability_analyzer.analyze(all_features, confidence)
        return str(info.get("trust_level", "MEDIUM"))

    selected_indices = select_demo_indices_by_trust_level(
        test_prompts,
        test_metrics["raw_probs"],
        trust_fn=_trust_tier_for_demo,
        n_per_level=2,
    )
    tier_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for j in selected_indices:
        tier_counts[_trust_tier_for_demo(j)] += 1
    print(
        f"Trust-tier picks — HIGH: {tier_counts['HIGH']}, MEDIUM: {tier_counts['MEDIUM']}, "
        f"LOW: {tier_counts['LOW']} (tiers from SignalReliabilityAnalyzer)"
    )

    trust_demo_prompt_max_chars = 4000
    trust_demo_records: List[Dict[str, Any]] = []
    test_labels_for_demo = test_metrics["labels"]
    calibrated_flag = args.calibration_method != "none"

    if args.trust_demo_samples_full_test:
        _full_iter = range(len(test_prompts))
        _full_iter = tqdm(_full_iter, desc="trust_demo_samples.json (full test)")
        for i in _full_iter:
            trust_demo_records.append(
                build_trust_sample_record(
                    i,
                    test_prompts,
                    test_probs_arr,
                    test_labels_for_demo,
                    test_sem,
                    test_unc,
                    int(args.stage),
                    calibrated_flag,
                    reliability_analyzer,
                    trust_demo_prompt_max_chars,
                    include_formatted_terminal=False,
                )
            )
        for i in selected_indices:
            rec_print = build_trust_sample_record(
                i,
                test_prompts,
                test_probs_arr,
                test_labels_for_demo,
                test_sem,
                test_unc,
                int(args.stage),
                calibrated_flag,
                reliability_analyzer,
                trust_demo_prompt_max_chars,
                include_formatted_terminal=True,
            )
            print(rec_print["formatted_terminal_output"])
    else:
        for i in selected_indices:
            rec = build_trust_sample_record(
                i,
                test_prompts,
                test_probs_arr,
                test_labels_for_demo,
                test_sem,
                test_unc,
                int(args.stage),
                calibrated_flag,
                reliability_analyzer,
                trust_demo_prompt_max_chars,
                include_formatted_terminal=True,
            )
            print(rec["formatted_terminal_output"])
            trust_demo_records.append(rec)

    trust_demo_meta = {
        "stage": int(args.stage),
        "calibration_method": args.calibration_method,
        "prompt_max_chars": trust_demo_prompt_max_chars,
        "n_per_level_requested": 2,
        "tier_pick_counts": tier_counts,
        "trust_tier_demo_indices": [int(x) for x in selected_indices],
    }
    if args.trust_demo_samples_full_test:
        trust_demo_meta.update({
            "description": "Full test split: every row includes trust tier + signal_reliability (no formatted_terminal_output per row).",
            "export_scope": "full_test_set",
            "num_samples": len(trust_demo_records),
            "includes_formatted_terminal_output": False,
        })
    else:
        trust_demo_meta.update({
            "description": "Qualitative trust-tier demo only; mirrors terminal Stage 3 trust demo order.",
            "export_scope": "trust_tier_demo_only",
            "num_samples": len(trust_demo_records),
            "includes_formatted_terminal_output": True,
        })

    export_trust_demo_samples_json(
        os.path.join(args.output_dir, "trust_demo_samples.json"),
        meta=trust_demo_meta,
        samples=trust_demo_records,
    )

    # Persist tokenizer
    tokenizer.save_pretrained(args.output_dir)

    print(f"\nDone. Artifacts directory: {args.output_dir}")
    print(f"Best Val AUROC: {best_auroc:.4f}")
    print(f"Test AUROC: {test_metrics['auroc']:.4f}")
    print(f"Test ECE: {test_metrics['ece']:.4f}")


def inference(args):
    """Forward pass helper used during train/eval loops."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Restore checkpoint
    ckpt = torch.load(os.path.join(args.model_path, "best.pt"), map_location="cpu")
    stage = ckpt.get("stage", 1)
    config = ckpt.get("config", {})

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    encoder = AutoModel.from_pretrained(ckpt.get("pretrained_model", "distilbert-base-uncased"))
    hidden_size = encoder.config.hidden_size

    if int(stage) == 1:
        model = SemanticAwareHallucinationPredictor(encoder, hidden_size, 4,
                                                     config.get("dropout", 0.1),
                                                     config.get("use_mean_pooling", False))
    else:
        model = EnhancedHallucinationPredictor(encoder, hidden_size, 4, 6,
                                               config.get("dropout", 0.1),
                                               config.get("use_mean_pooling", False))

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Restore calibrator weights
    calibrator = None
    cal_path = os.path.join(args.model_path, "calibrator.pt")
    if os.path.exists(cal_path) and args.use_calibration:
        if args.calibration_method == "temperature":
            calibrator = TemperatureScaling()
        elif args.calibration_method == "platt":
            calibrator = PlattScaling()
        calibrator.load_state_dict(torch.load(cal_path, map_location="cpu"))
        calibrator.to(device)

    # Single-example inference
    if args.input:
        enc = tokenizer(args.input, truncation=True, max_length=config.get("max_length", 256),
                       return_tensors="pt")

        # Semantic telemetry
        semantic_analyzer = SemanticAnalyzer(ckpt.get("pretrained_model", "distilbert-base-uncased"), device)
        sem_feat = compute_semantic_features(args.input, semantic_analyzer)

        item = {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "label": torch.tensor(0),
            "semantic": {
                "semantic_similarity": sem_feat['semantic_similarity'],
                "entity_overlap_ratio": sem_feat['entity_overlap_ratio'],
                "knowledge_overlap": sem_feat['knowledge_overlap'],
                "entity_coverage": sem_feat['entity_coverage'],
            },
        }
        if int(stage) >= 2:
            unc = compute_uncertainty_features(args.input)
            item["uncertainty"] = {
                k: torch.tensor(v, dtype=torch.float32)
                for k, v in unc.items()
                if k in (
                    "perplexity", "token_entropy", "answer_length",
                    "answer_char_length", "avg_confidence", "sequence_entropy",
                )
            }
        batch = collate_fn([item])

        probs, _ = predict(model, batch, device, stage, calibrator)
        prob = probs.item()
        confidence = 1 - prob

        ans_start = args.input.lower().find("answer:")
        answer = args.input[ans_start+7:].strip() if ans_start != -1 else args.input

        # ===== Stage 2.5: reliability reasoning =====
        reliability_analyzer = SignalReliabilityAnalyzer()
        
        # Merge cue dictionaries
        all_features = {
            'semantic_similarity': sem_feat['semantic_similarity'],
            'entity_overlap_ratio': sem_feat['entity_overlap_ratio'],
            'knowledge_overlap': sem_feat['knowledge_overlap'],
            'entity_coverage': sem_feat['entity_coverage'],
        }
        if int(stage) >= 2:
            all_features.update({
                'perplexity': unc['perplexity'],
                'token_entropy': unc['token_entropy'],
                'answer_length': unc['answer_length'],
            })
        
        # Analyzer.forward-equivalent
        reliability_info = reliability_analyzer.analyze(all_features, confidence)
        
        print(format_answer_output(answer, confidence, prob,
                                   is_calibrated=calibrator is not None,
                                   reliability_info=reliability_info))

    else:
        print("Provide --input text when running inference mode.")


def main():
    parser = argparse.ArgumentParser(description="Failure-aware hallucination detector (semantic stage 1, uncertainty stage 2, calibration stage 3)")

    # Mode
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "inference"])

    # Stage selector
    parser.add_argument("--stage", type=int, default=3, choices=[1, 2, 3],
                        help="1=semantic cues only, 2=add uncertainty branch, 3=full pipeline + calibration")

    # Paths
    parser.add_argument("--data_dir", type=str, default="Data")
    parser.add_argument("--output_dir", type=str, default="outputs_failure_aware_semantic_signal_reliability")
    parser.add_argument("--model_path", type=str, default="outputs_failure_aware_semantic_signal_reliability",
                        help="Checkpoint directory used during inference.")

    # Data
    parser.add_argument("--dataset_type", type=str, default="all",
                        choices=["all", "qa", "dialogue", "summarization", "general"])
    parser.add_argument("--exclude_task", type=str, default="",
                        choices=["", "qa", "dialogue", "summarization", "general"],
                        help="Hold out one sub-task for OOD generalization testing "
                             "(only applies when --dataset_type=all). "
                             "e.g. --exclude_task summarization trains on qa+dialogue+general.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap samples per shard for smoke tests.")

    # Model
    parser.add_argument("--pretrained_model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--use_mean_pooling", action="store_true")
    parser.add_argument("--freeze_encoder", action="store_true")

    # Optimization
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
    parser.add_argument(
        "--max_failure_cases_per_type",
        type=int,
        default=25,
        help="Rows per category in failure_cases.json (FP / FN). Use 0 to export all test-set errors.",
    )
    parser.add_argument(
        "--trust_demo_samples_full_test",
        action="store_true",
        help="Write trust_demo_samples.json for every test-split example (omits formatted_terminal_output per row). Terminal trust demo unchanged.",
    )

    # Single-example inference
    parser.add_argument("--input", type=str, default=None, help="Prompt string for inference mode.")
    parser.add_argument("--use_calibration", action="store_true", default=True,
                        help="Apply calibration during inference.")
    parser.add_argument("--no_calibration", dest="use_calibration", action="store_false",
                        help="Skip calibration transform.")

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        inference(args)


if __name__ == "__main__":
    main()
