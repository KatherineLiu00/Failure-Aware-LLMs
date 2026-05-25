"""
Convert HaluEval evaluation results into compare_baseline format.
"""

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


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


def normalize_judgement(raw: str) -> str:
    text = (raw or "").strip().lower()
    if text == "yes":
        return "yes"
    if text == "no":
        return "no"
    return "failed"


def parse_labels_and_probs(result_path: str) -> Tuple[List[int], List[float], int]:
    labels: List[int] = []
    probs: List[float] = []
    failed_count = 0

    with open(result_path, "r", encoding="utf-8") as f:
        for line in f:
            row = line.strip()
            if not row:
                continue
            item = json.loads(row)
            gt = str(item.get("ground_truth", "")).strip().lower()
            judgement = normalize_judgement(str(item.get("judgement", "")))

            if gt not in {"yes", "no"}:
                continue

            label = 1 if gt == "yes" else 0
            if judgement == "yes":
                prob = 1.0
            elif judgement == "no":
                prob = 0.0
            else:
                # Keep uncertain output as neutral probability.
                prob = 0.5
                failed_count += 1

            labels.append(label)
            probs.append(prob)

    return labels, probs, failed_count


QAAlignKey = Tuple[str, str, str]


def build_qa_answer_judge_lookup(
    result_path: str,
) -> Tuple[Dict[QAAlignKey, Tuple[float, bool]], int]:
    """
    Map (knowledge, question, answer) -> (prob for label=hallucination, judgement_parse_failed).
    Duplicate triples in jsonl: last line wins.
    """
    lookup: Dict[QAAlignKey, Tuple[float, bool]] = {}
    duplicates = 0
    with open(result_path, "r", encoding="utf-8") as f:
        for line in f:
            row = line.strip()
            if not row:
                continue
            item = json.loads(row)
            key = (
                str(item.get("knowledge", "")).strip(),
                str(item.get("question", "")).strip(),
                str(item.get("answer", "")).strip(),
            )
            if key in lookup:
                duplicates += 1
            judgement = normalize_judgement(str(item.get("judgement", "")))
            if judgement == "yes":
                lookup[key] = (1.0, False)
            elif judgement == "no":
                lookup[key] = (0.0, False)
            else:
                lookup[key] = (0.5, True)
    return lookup, duplicates


def parse_qa_training_aligned(result_path: str, qa_data_path: str) -> Tuple[List[int], List[float], int]:
    """
    Expand judge jsonl to the same ordering as train scripts: each qa row ->
      (right_answer, label 0), (hallucinated_answer, label 1).
    Rows missing from qa_gpt-3.5-turbo_result-style files get prob 0.5 (coverage gap).
    """
    lookup, dup_keys = build_qa_answer_judge_lookup(result_path)
    labels: List[int] = []
    probs: List[float] = []
    failed_count = 0

    n_qa_rows = 0
    with open(qa_data_path, "r", encoding="utf-8") as f:
        for line in f:
            row = line.strip()
            if not row:
                continue
            obj = json.loads(row)
            n_qa_rows += 1
            k = str(obj.get("knowledge", "")).strip()
            q = str(obj.get("question", "")).strip()
            right_a = str(obj.get("right_answer", "")).strip()
            hall_a = str(obj.get("hallucinated_answer", "")).strip()

            for answer_text, lab in ((right_a, 0), (hall_a, 1)):
                key = (k, q, answer_text)
                if key not in lookup:
                    probs.append(0.5)
                    labels.append(lab)
                    failed_count += 1
                else:
                    prob, jfail = lookup[key]
                    probs.append(prob)
                    labels.append(lab)
                    if jfail:
                        failed_count += 1

    print(
        f"QA aligned to train order: qa_data rows={n_qa_rows} -> {len(labels)} samples "
        f"| keys in judge jsonl={len(lookup)} dup={dup_keys} "
        f"| neutral/absent+pseudo-failed count={failed_count}"
    )
    return labels, probs, failed_count


def _safe_ratio(num: float, den: float) -> float:
    return 0.0 if den <= 0 else num / den


def _token_set(text: str) -> set:
    return {w for w in str(text).lower().split() if w}


def _hallucination_prob(knowledge: str, question_or_context: str, answer: str) -> float:
    """
    Simple local heuristic baseline:
    lower overlap with knowledge/context => higher hallucination probability.
    """
    ans_tokens = _token_set(answer)
    if not ans_tokens:
        return 0.6

    know_tokens = _token_set(knowledge)
    ctx_tokens = _token_set(question_or_context)
    overlap_k = _safe_ratio(len(ans_tokens & know_tokens), len(ans_tokens))
    overlap_c = _safe_ratio(len(ans_tokens & ctx_tokens), len(ans_tokens))

    base = 1.0 - (0.7 * overlap_k + 0.3 * overlap_c)
    return float(min(1.0, max(0.0, base)))


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = line.strip()
            if not row:
                continue
            yield json.loads(row)


def parse_from_halueval_data(data_dir: str) -> Tuple[List[int], List[float], Dict[str, int]]:
    labels: List[int] = []
    probs: List[float] = []
    stats: Dict[str, int] = {"qa": 0, "dialogue": 0, "summarization": 0, "general": 0}

    qa_path = os.path.join(data_dir, "qa_data.json")
    if os.path.exists(qa_path):
        for obj in _iter_jsonl(qa_path):
            knowledge = obj.get("knowledge", "")
            question = obj.get("question", "")
            right_answer = obj.get("right_answer", "")
            hallucinated_answer = obj.get("hallucinated_answer", "")
            probs.append(_hallucination_prob(knowledge, question, right_answer))
            labels.append(0)
            probs.append(_hallucination_prob(knowledge, question, hallucinated_answer))
            labels.append(1)
            stats["qa"] += 2

    dialogue_path = os.path.join(data_dir, "dialogue_data.json")
    if os.path.exists(dialogue_path):
        for obj in _iter_jsonl(dialogue_path):
            knowledge = obj.get("knowledge", "")
            history = obj.get("dialogue_history", "")
            right_response = obj.get("right_response", "")
            hallucinated_response = obj.get("hallucinated_response", "")
            probs.append(_hallucination_prob(knowledge, history, right_response))
            labels.append(0)
            probs.append(_hallucination_prob(knowledge, history, hallucinated_response))
            labels.append(1)
            stats["dialogue"] += 2

    sum_path = os.path.join(data_dir, "summarization_data.json")
    if os.path.exists(sum_path):
        for obj in _iter_jsonl(sum_path):
            document = obj.get("document", "")
            right_summary = obj.get("right_summary", "")
            hallucinated_summary = obj.get("hallucinated_summary", "")
            probs.append(_hallucination_prob(document, "", right_summary))
            labels.append(0)
            probs.append(_hallucination_prob(document, "", hallucinated_summary))
            labels.append(1)
            stats["summarization"] += 2

    general_path = os.path.join(data_dir, "general_data.json")
    if os.path.exists(general_path):
        for obj in _iter_jsonl(general_path):
            query = obj.get("user_query", "")
            response = obj.get("chatgpt_response", "")
            hall = str(obj.get("hallucination_label", obj.get("hallucination", "no"))).strip().lower()
            label = 1 if hall == "yes" else 0
            probs.append(_hallucination_prob("", query, response))
            labels.append(label)
            stats["general"] += 1

    return labels, probs, stats


def split_stratified(
    y: List[int], test_ratio: float, val_ratio: float, seed: int
) -> Tuple[List[int], List[int], List[int]]:
    """
    Keep identical split behavior with train_failure_aware.py.
    """
    rng = random.Random(seed)
    idx0 = [i for i, yi in enumerate(y) if yi == 0]
    idx1 = [i for i, yi in enumerate(y) if yi == 1]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    def split(idx_list: List[int]) -> Tuple[List[int], List[int], List[int]]:
        n = len(idx_list)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        n_train = n - n_test - n_val
        return idx_list[:n_train], idx_list[n_train:n_train + n_val], idx_list[n_train + n_val:]

    tr0, va0, te0 = split(idx0)
    tr1, va1, te1 = split(idx1)

    train_idx = tr0 + tr1
    val_idx = va0 + va1
    test_idx = te0 + te1

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def select_by_indices(values: List[float], indices: List[int]) -> List[float]:
    return [values[i] for i in indices]


def compute_metrics(labels: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    metrics = {}
    if len(np.unique(labels)) > 1:
        metrics["auroc"] = float(roc_auc_score(labels, probs))
        metrics["aupr"] = float(average_precision_score(labels, probs))
    else:
        metrics["auroc"] = float("nan")
        metrics["aupr"] = float("nan")
    metrics["ece"] = compute_ece(probs, labels)
    metrics["ece_10"] = compute_ece(probs, labels, 10)
    return metrics


def auto_discover_result_file(result_root: str, dataset_type: str = "all") -> str:
    """
    Auto-find HaluEval evaluate.py result file.
    Priority: qa > dialogue > summarization, then alphabetical.
    """
    root_dir = result_root
    if not os.path.isdir(root_dir):
        raise ValueError(f"Cannot find '{root_dir}' directory in current working directory.")

    matches: List[str] = []
    for current_root, _, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith("_results.json") or filename.endswith("_result.json"):
                matches.append(os.path.join(current_root, filename))

    if not matches:
        raise ValueError(
            "Cannot find any '*_result.json' or '*_results.json' under result_root. "
            "Please run HaluEval/evaluate.py first, or pass --result_file explicitly."
        )

    if dataset_type == "qa":
        qa_only = [m for m in matches if "qa" in os.path.basename(m).lower()]
        if qa_only:
            matches = qa_only

    priority = {"qa": 0, "dialogue": 1, "summarization": 2}

    def sort_key(path: str):
        parts = path.replace("\\", "/").split("/")
        task = parts[1] if len(parts) > 1 else ""
        return (priority.get(task, 9), path)

    matches.sort(key=sort_key)
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description="Build HaluEval baseline for compare_baseline.py")
    parser.add_argument(
        "--result_file",
        default=None,
        help="Path to HaluEval evaluate.py output jsonl file (e.g., HaluEval/qa/qa_davinci_results.json)",
    )
    parser.add_argument("--result_root", default="HaluEval-Baseline", help="Directory to scan for *_results.json")
    parser.add_argument("--data_dir", default="HaluEval-Data", help="Fallback data directory when no result file exists")
    parser.add_argument("--output_dir", default="outputs_halueval_baseline", help="Output directory")
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="all",
        choices=["all", "qa"],
        help=(
            "'qa': expand qa_data.json to match train FAILURE_aware loaders, join judge probs by triple key; "
            "required with partial qa_*_result.json for fair qa_compare_baseline splits."
        ),
    )
    parser.add_argument(
        "--qa_data_path",
        type=str,
        default=None,
        help="Overrides {data_dir}/qa_data.json when --dataset_type qa.",
    )
    parser.add_argument("--name", default="HaluEval Baseline", help="Model name for metadata")
    parser.add_argument(
        "--no_fallback",
        action="store_true",
        help="Do not fallback to HaluEval-Data heuristic baseline when no result file is found.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stratified split")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument(
        "--use_test_split_only",
        dest="use_test_split_only",
        action="store_true",
        default=True,
        help="Use only the stratified test split (aligned with training scripts)",
    )
    parser.add_argument(
        "--use_full_data",
        dest="use_test_split_only",
        action="store_false",
        help="Disable split filtering and evaluate on all samples",
    )
    args = parser.parse_args()

    source_desc = ""
    qa_path_opt = args.qa_data_path or os.path.join(args.data_dir, "qa_data.json")

    if args.result_file:
        result_file = args.result_file
        if args.dataset_type == "qa":
            if not os.path.isfile(qa_path_opt):
                raise FileNotFoundError(f"--dataset_type qa requires qa_data.json at {qa_path_opt}")
            labels, probs, failed_count = parse_qa_training_aligned(result_file, qa_path_opt)
            source_desc = f"result_file:{result_file}|qa_aligned:{qa_path_opt}"
        else:
            labels, probs, failed_count = parse_labels_and_probs(result_file)
            source_desc = f"result_file:{result_file}"
    else:
        try:
            result_file = auto_discover_result_file(args.result_root, args.dataset_type)
            print(f"Auto-detected result_file: {result_file}")
            if args.dataset_type == "qa":
                if not os.path.isfile(qa_path_opt):
                    raise FileNotFoundError(f"--dataset_type qa requires qa_data.json at {qa_path_opt}")
                labels, probs, failed_count = parse_qa_training_aligned(result_file, qa_path_opt)
                source_desc = f"result_file:{result_file}|qa_aligned:{qa_path_opt}"
            else:
                labels, probs, failed_count = parse_labels_and_probs(result_file)
                source_desc = f"result_file:{result_file}"
        except ValueError:
            if args.no_fallback:
                raise ValueError(
                    "No evaluation result file found and --no_fallback is set. "
                    "Please pass --result_file, e.g. "
                    "'HaluEval-Baseline/HaluEval-Data/qa_gpt-3.5-turbo_result.json'."
                )
            labels, probs, data_stats = parse_from_halueval_data(args.data_dir)
            failed_count = 0
            source_desc = f"data_dir:{args.data_dir}"
            print(f"No *_results.json found, fallback to local dataset: {args.data_dir}")
            print(f"Loaded samples by task: {data_stats}")

    if not labels:
        raise ValueError("No valid samples parsed. Provide --result_file or check --data_dir.")

    split_info = {
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "val_ratio": args.val_ratio,
        "mode": "all_data",
    }
    if args.use_test_split_only:
        train_idx, val_idx, test_idx = split_stratified(labels, args.test_ratio, args.val_ratio, args.seed)
        labels = select_by_indices(labels, test_idx)
        probs = select_by_indices(probs, test_idx)
        split_info = {
            "seed": args.seed,
            "test_ratio": args.test_ratio,
            "val_ratio": args.val_ratio,
            "mode": "test_only",
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
        }
        print(
            "Using stratified split: "
            f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    labels_np = np.array(labels, dtype=np.int64)
    probs_np = np.array(probs, dtype=np.float64)
    metrics = compute_metrics(labels_np, probs_np)

    with open(os.path.join(args.output_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump({"labels": labels, "probs": probs}, f)

    payload = {
        "model": args.name,
        "source": source_desc,
        "num_samples": len(labels),
        "failed_judgement_count": failed_count,
        "split": split_info,
        "metrics": metrics,
    }
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Saved predictions to: {os.path.join(args.output_dir, 'predictions.json')}")
    print(f"Saved metrics to: {os.path.join(args.output_dir, 'test_metrics.json')}")
    if args.use_test_split_only:
        print(
            f"Samples in output: {len(labels)} (test split only) | "
            f"full-corpus neutral/missing-judge count before split: {failed_count}"
        )
    else:
        print(f"Samples: {len(labels)} | failed judgements mapped to 0.5: {failed_count}")


if __name__ == "__main__":
    main()

