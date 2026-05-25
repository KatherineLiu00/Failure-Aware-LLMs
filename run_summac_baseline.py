"""
Run published SummaC (Laban et al.) on HaluEval-Data with the same split as
train_failure_aware_semantic_signal_reliability.py, and write predictions.json
for compare_baseline.py.

Install: pip install summac

SummaC returns a *consistency* score (higher = more supported / more faithful).
We map to P(hallucination) for alignment with training exports:
  - Val-set min–max of consistency, then prob_hallucination = 1 - norm(consistency)
  (calibration uses only the validation split to avoid test leakage).

Category (for papers): "Published NLI-based document–summary consistency (SummaC)".
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

# Same data order & split as proposed training
from train_failure_aware_semantic_signal_reliability import load_all_data, split_stratified


def document_and_summary_from_prompt(prompt: str) -> Tuple[str, str]:
    """
    Map HaluEval prompt lines to (document, summary) for SummaC.
    document = evidence / source context; summary = answer to score.
    """
    p = prompt.strip()
    low = p.lower()
    if low.startswith("document:") and "summary:" in low:
        si = p.lower().find("summary:")
        doc = p[len("Document:") : si].strip()
        summ = p[si + len("Summary:") :].strip()
        return doc, summ
    if "knowledge:" in low and "question:" in low and "answer:" in low:
        qi = p.lower().find("question:")
        ai = p.lower().find("answer:")
        if qi >= 0 and ai > qi:
            k = p[p.lower().find("knowledge:") + len("Knowledge:") : qi].strip()
            qtext = p[qi + len("Question:") : ai].strip()
            a = p[ai + len("Answer:") :].strip()
            return f"{k}\n{qtext}".strip(), a
    if "knowledge:" in low and "dialogue:" in low and "response:" in low:
        di = p.lower().find("dialogue:")
        ri = p.lower().find("response:")
        if di >= 0 and ri > di:
            k = p[p.lower().find("knowledge:") + len("Knowledge:") : di].strip()
            dtext = p[di + len("Dialogue:") : ri].strip()
            r = p[ri + len("Response:") :].strip()
            return f"{k}\n{dtext}".strip(), r
    if "query:" in low and "response:" in low:
        qi = p.lower().find("query:")
        ri = p.lower().find("response:")
        if qi >= 0 and ri > qi:
            q = p[qi + len("Query:") : ri].strip()
            r = p[ri + len("Response:") :].strip()
            return q, r
    return "", p


def compute_ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def main():
    parser = argparse.ArgumentParser(description="SummaC baseline -> predictions.json (HaluEval all tasks)")
    parser.add_argument("--data_dir", type=str, default="HaluEval-Data")
    parser.add_argument("--dataset_type", type=str, default="all")
    parser.add_argument("--output_dir", type=str, default="outputs_summac_baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=None, help="cpu | cuda | None=auto")
    parser.add_argument(
        "--granularity",
        type=str,
        default="sentence",
        choices=["sentence", "document"],
        help="SummaCZS granularity",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="vitc",
        help="SummaCZS model_name (see summac docs)",
    )
    args = parser.parse_args()

    device = args.device
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        from summac.model_summac import SummaCZS
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: pip install summac\n"
            "Also ensure torch / transformers are installed."
        ) from e

    os.makedirs(args.output_dir, exist_ok=True)

    prompts, labels, data_stats = load_all_data(args.data_dir, args.dataset_type)
    train_idx, val_idx, test_idx = split_stratified(labels, args.test_ratio, args.val_ratio, args.seed)

    print(f"Loaded {len(prompts)} samples | test split size = {len(test_idx)}")
    model = SummaCZS(granularity=args.granularity, model_name=args.model_name, device=device)

    def scores_for_indices(indices: List[int], desc: str) -> List[float]:
        out: List[float] = []
        for i in tqdm(indices, desc=desc):
            doc, summ = document_and_summary_from_prompt(prompts[i])
            if not summ.strip():
                out.append(0.0)
                continue
            if not doc.strip():
                doc = summ[:512]
            try:
                batch = model.score([doc], [summ])
                sc = float(batch["scores"][0])
            except Exception:
                sc = 0.0
            out.append(sc)
        return out

    print("Scoring validation split (for calibration bounds)...")
    val_scores = np.array(scores_for_indices(val_idx, "SummaC val"), dtype=np.float64)
    print("Scoring test split...")
    test_scores = np.array(scores_for_indices(test_idx, "SummaC test"), dtype=np.float64)

    vmin, vmax = float(val_scores.min()), float(val_scores.max())
    denom = vmax - vmin if (vmax - vmin) > 1e-8 else 1.0

    def normalize_consistency(sc: np.ndarray) -> np.ndarray:
        return (sc - vmin) / denom

    val_norm = np.clip(normalize_consistency(val_scores), 0.0, 1.0)
    test_norm = np.clip(normalize_consistency(test_scores), 0.0, 1.0)

    # Higher consistency -> lower hallucination probability
    val_probs = 1.0 - val_norm
    test_probs = (1.0 - test_norm).tolist()

    test_labels = [int(labels[i]) for i in test_idx]

    predictions = {"probs": test_probs, "labels": test_labels}
    with open(os.path.join(args.output_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(predictions, f)

    val_labels_arr = np.array([labels[i] for i in val_idx], dtype=np.int64)
    val_probs_arr = np.array(val_probs, dtype=np.float64)
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_val = val_labels_arr
    p_val = val_probs_arr
    if len(np.unique(y_val)) > 1:
        v_auroc = float(roc_auc_score(y_val, p_val))
        v_ap = float(average_precision_score(y_val, p_val))
    else:
        v_auroc, v_ap = float("nan"), float("nan")

    y_te = np.array(test_labels, dtype=np.int64)
    p_te = np.array(test_probs, dtype=np.float64)
    te_auroc = float(roc_auc_score(y_te, p_te))
    te_ap = float(average_precision_score(y_te, p_te))
    te_ece = compute_ece(p_te, y_te)

    metrics_wrap = {
        "baseline_category": "published_consistency_summac",
        "baseline_description": "SummaC (NLI-based document-summary consistency); scores inverted via val min-max to P(hallucination)",
        "summac": {"model_name": args.model_name, "granularity": args.granularity, "device": device},
        "calibration": "val_minmax_consistency_to_prob_halluc",
        "split": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "data_stats": data_stats,
        "metrics": {
            "auroc": te_auroc,
            "aupr": te_ap,
            "ece": te_ece,
            "auroc_val": v_auroc,
            "aupr_val": v_ap,
        },
    }
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_wrap, f, indent=2, ensure_ascii=False)

    print(f"Wrote {args.output_dir}/predictions.json and test_metrics.json")
    print(f"Test AUROC={te_auroc:.4f} AUPR={te_ap:.4f} ECE={te_ece:.4f}")


if __name__ == "__main__":
    main()
