"""
Appendix (Scheme 2): QA-aligned comparison only.

Plots Baseline vs Proposed trained on QA split alongside HaluEval official-style
LLM-as-judge probabilities from build_halueval_baseline.py — same stratified split
(seed/test_ratio/val_ratio) as training for fair QA-only curves.

Do not mix these curves with compare_baseline.py (all-task main results).
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

BASELINE_CATEGORIES = {
    "HaluEval Baseline (LLM-as-judge reference)": (
        "HaluEval-style LLM-as-judge probabilities (reference line; not same architecture as local models)"
    ),
    "Baseline (without signal reliability)": (
        "Learned detector trained on QA-only split (no signal-reliability branch)"
    ),
    "Proposed (with signal reliability)": (
        "Our method trained on QA-only split (semantic + uncertainty + signal reliability)"
    ),
    "SummaC (published consistency)": (
        "Published SummaC consistency baseline on QA-only split (same seed/split as training; see run_summac_baseline.py)"
    ),
}


def load_predictions(pred_path: str) -> dict:
    with open(pred_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_metrics(metrics_path: str) -> dict:
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _curve_spec():
    return {
        "HaluEval Baseline (LLM-as-judge reference)": {"color": "m", "fill_alpha": 0.08},
        "Baseline (without signal reliability)": {"color": "b", "fill_alpha": 0.10},
        "Proposed (with signal reliability)": {"color": "g", "fill_alpha": 0.20},
        "SummaC (published consistency)": {"color": "darkorange", "fill_alpha": 0.12},
    }


def plot_pr_curve_comparison(model_preds: dict, output_path: str):
    fig, ax = plt.subplots(figsize=(8, 8))

    specs = _curve_spec()
    random_ref = None
    for name, preds in model_preds.items():
        probs = np.array(preds["probs"])
        labels = np.array(preds["labels"])
        ap = average_precision_score(labels, probs)

        precision, recall, _ = precision_recall_curve(labels, probs)
        spec = specs.get(name, {"color": "k", "fill_alpha": 0.08})
        ax.plot(
            recall,
            precision,
            color=spec["color"],
            linestyle="-",
            linewidth=2,
            label=f"{name} (AP = {ap:.4f})",
        )
        ax.fill_between(recall, precision, alpha=spec["fill_alpha"], color=spec["color"])
        if random_ref is None:
            random_ref = labels.mean()

    random_ap = 0.0 if random_ref is None else float(random_ref)
    ax.axhline(y=random_ap, color="r", linestyle="--", alpha=0.7, label=f"Random (AP = {random_ap:.4f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Appendix — PR (QA-only, judge reference)", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"PR curve comparison saved: {output_path}")


def plot_roc_curve_comparison(model_preds: dict, output_path: str):
    fig, ax = plt.subplots(figsize=(8, 8))

    specs = _curve_spec()
    for name, preds in model_preds.items():
        probs = np.array(preds["probs"])
        labels = np.array(preds["labels"])
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)

        spec = specs.get(name, {"color": "k", "fill_alpha": 0.08})
        ax.plot(
            fpr,
            tpr,
            color=spec["color"],
            linestyle="-",
            linewidth=2,
            label=f"{name} (AUC = {roc_auc:.4f})",
        )
        ax.fill_between(fpr, tpr, alpha=spec["fill_alpha"], color=spec["color"])

    ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="Random (AUC = 0.5)")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Appendix — ROC (QA-only, judge reference)", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC curve comparison saved: {output_path}")


def _calibration_bin_points_quantile(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int,
    min_bin_samples: int,
):
    """Quantile bins → (mean_pred, frac_pos, n_per_bin) as float lists."""
    order = np.argsort(probs)
    probs_s = probs[order]
    labels_s = labels[order]
    bin_indices = np.array_split(np.arange(len(probs_s)), n_bins)

    mean_pred, frac_pos, weights = [], [], []
    for bi in bin_indices:
        n = len(bi)
        if n < min_bin_samples:
            continue
        p_hat = float(labels_s[bi].mean())
        mean_pred.append(float(probs_s[bi].mean()))
        frac_pos.append(p_hat)
        weights.append(float(n))
    return mean_pred, frac_pos, weights


def plot_calibration_curve_comparison(
    model_preds: dict,
    all_metrics: dict,
    output_path: str,
    title: str = "Appendix — Calibration (QA-only, judge reference)",
    n_bins: int = 5,
    min_bin_samples: int = 50,
):
    """
    Single-panel reliability diagram.
    Quantile binning gives stable bin sizes; a weighted isotonic regression on
    bin centers removes residual zig-zag from sampling noise while preserving
    the overall trend. Raw bin accuracies are shown as faint scatter markers.
    No confidence band is drawn. Legend ECE is from the full test set (metrics files).
    """
    specs = _curve_spec()
    linestyles = ["-", "--", "-.", ":"]
    markers    = ["o", "s", "^", "D"]
    names = list(model_preds.keys())

    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, name in enumerate(names):
        preds  = model_preds[name]
        probs  = np.array(preds["probs"],  dtype=np.float64)
        labels = np.array(preds["labels"], dtype=np.float64)
        color  = specs.get(name, {"color": "k"})["color"]
        ls     = linestyles[idx % len(linestyles)]
        mk     = markers[idx % len(markers)]

        mean_pred, frac_pos, weights = _calibration_bin_points_quantile(
            probs, labels, n_bins=n_bins, min_bin_samples=min_bin_samples
        )

        ece_val = all_metrics.get(name, {}).get("metrics", {}).get("ece", float("nan"))
        short   = name.split("(")[0].strip()
        ece_str = f"ECE={ece_val:.4f}" if not np.isnan(ece_val) else "ECE=N/A"
        label   = f"{short} ({ece_str})"

        if mean_pred:
            m_arr = np.array(mean_pred, dtype=np.float64)
            f_arr = np.array(frac_pos, dtype=np.float64)
            w_arr = np.array(weights, dtype=np.float64)
            order = np.argsort(m_arr)
            m_arr, f_arr, w_arr = m_arr[order], f_arr[order], w_arr[order]

            if len(m_arr) >= 2:
                iso = IsotonicRegression(
                    y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True
                )
                iso.fit(m_arr, f_arr, sample_weight=w_arr)
                f_smooth = iso.predict(m_arr)
            else:
                f_smooth = f_arr

            ax.plot(
                m_arr, f_smooth, color=color, linestyle=ls, marker=mk, markersize=7,
                linewidth=2, label=label, zorder=3,
            )
            ax.scatter(
                m_arr, f_arr, color=color, s=28, alpha=0.35,
                edgecolors="none", zorder=4,
            )
        else:
            ax.plot([], [], color=color, linestyle=ls, marker=mk,
                    markersize=7, linewidth=2, label=f"{label} [no data]")

    ax.plot([0, 1], [0, 1], color="black", linestyle="--",
            linewidth=1.5, label="Perfect calibration", zorder=1)
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.92, handlelength=2.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.text(
        0.98, 0.03,
        f"Quantile bins (n={n_bins}), min {min_bin_samples} samples/bin · "
        f"line = isotonic-smoothed · dots = raw bin rates (no CI band)",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7.5, color="grey",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Calibration curve comparison saved: {output_path}")


def print_metrics_comparison(all_metrics: dict, all_preds: dict):
    print("\n" + "=" * 70)
    print("Appendix — QA-aligned comparison (includes LLM-as-judge reference)")
    print("=" * 70)
    print("\nBaseline categories:")
    for name in all_preds:
        print(f"  • {name}: {BASELINE_CATEGORIES.get(name, '(custom)')}")

    print(f"{'Model':<42} {'AUROC':<10} {'AUPR(AP)':<10} {'ECE':<10}")
    print("-" * 70)

    for name, preds in all_preds.items():
        labels = np.array(preds["labels"])
        probs = np.array(preds["probs"])
        fpr, tpr, _ = roc_curve(labels, probs)
        model_auc = auc(fpr, tpr)
        model_ap = average_precision_score(labels, probs)
        model_ece = all_metrics.get(name, {}).get("metrics", {}).get("ece", float("nan"))
        ece_str = f"{model_ece:.4f}" if not np.isnan(model_ece) else "N/A"
        print(f"{name:<42} {model_auc:<10.4f} {model_ap:<10.4f} {ece_str:<10}")

    print("=" * 70)


def _load_one_model(model_name: str, model_dir: str):
    pred_path = os.path.join(model_dir, "predictions.json")
    metrics_path = os.path.join(model_dir, "test_metrics.json")

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"{model_name} predictions not found: {pred_path}")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"{model_name} metrics not found: {metrics_path}")
    return load_predictions(pred_path), load_metrics(metrics_path)


def main():
    parser = argparse.ArgumentParser(
        description="Appendix: QA-only PR/ROC vs HaluEval judge baseline (Scheme 2)."
    )
    parser.add_argument("--baseline_dir", default="./outputs_failure_aware_qa")
    parser.add_argument("--proposed_dir", default="./outputs_failure_aware_semantic_signal_reliability_qa")
    parser.add_argument("--halueval_dir", default="./outputs_halueval_baseline_qa")
    parser.add_argument("--summac_dir", default="./outputs_summac_baseline_qa")
    parser.add_argument("--output_dir", default="./outputs_comparison_qa")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_preds = {}
    all_metrics = {}

    model_entries = [
        ("HaluEval Baseline (LLM-as-judge reference)", args.halueval_dir),
        ("Baseline (without signal reliability)", args.baseline_dir),
        ("Proposed (with signal reliability)", args.proposed_dir),
    ]
    summac_pred = os.path.join(args.summac_dir, "predictions.json")
    summac_met = os.path.join(args.summac_dir, "test_metrics.json")
    if os.path.isfile(summac_pred) and os.path.isfile(summac_met):
        model_entries.append(("SummaC (published consistency)", args.summac_dir))
        print(f"Including SummaC from {args.summac_dir}")
    else:
        print(
            f"No QA-only SummaC found at {args.summac_dir}; skipping. Run:\n"
            f"  python run_summac_baseline.py --dataset_type qa --output_dir {args.summac_dir}"
        )

    proposed_name = "Proposed (with signal reliability)"
    preds_p, metrics_p = _load_one_model(proposed_name, args.proposed_dir)
    ref_n = len(preds_p["labels"])
    all_preds[proposed_name] = preds_p
    all_metrics[proposed_name] = metrics_p
    print(f"Loaded {proposed_name}: {ref_n} samples (canonical QA test size) from {args.proposed_dir}")

    for model_name, model_dir in model_entries:
        if model_name == proposed_name:
            continue
        preds, metrics = _load_one_model(model_name, model_dir)
        n = len(preds["labels"])
        if n != ref_n:
            raise ValueError(
                f"Sample count mismatch for {model_name}: got {n}, expected {ref_n}. "
                "Rebuild QA HaluEval baseline (expand to 20k + same split as train) with:\n"
                "  python HaluEval-Baseline/build_halueval_baseline.py --dataset_type qa \\\n"
                "    --result_file HaluEval-Baseline/qa_gpt-3.5-turbo_result.json \\\n"
                "    --data_dir HaluEval-Data --output_dir ./outputs_halueval_baseline_qa"
            )
        all_preds[model_name] = preds
        all_metrics[model_name] = metrics
        print(f"Loaded {model_name}: {n} samples from {model_dir}")

    order = [
        "HaluEval Baseline (LLM-as-judge reference)",
        "Baseline (without signal reliability)",
        "Proposed (with signal reliability)",
    ]
    if "SummaC (published consistency)" in all_preds:
        order.append("SummaC (published consistency)")
    all_preds = {k: all_preds[k] for k in order}
    all_metrics = {k: all_metrics[k] for k in order}

    print(
        "HaluEval line = LLM-as-judge reference (not same architecture as local detectors).\n"
        "Main paper figure: python compare_baseline.py (all-task Baseline vs Proposed only)."
    )

    plot_pr_curve_comparison(all_preds, os.path.join(args.output_dir, "pr_curve_comparison.png"))
    plot_roc_curve_comparison(all_preds, os.path.join(args.output_dir, "roc_curve_comparison.png"))
    plot_calibration_curve_comparison(
        all_preds,
        all_metrics,
        os.path.join(args.output_dir, "calibration_comparison.png"),
        title="Appendix — Calibration (QA-only, judge reference)",
    )
    print_metrics_comparison(all_metrics, all_preds)
    print(f"\nAppendix QA comparison outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

