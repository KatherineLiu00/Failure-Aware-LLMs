"""
Main result comparison (Scheme 2): local trained models on all HaluEval tasks.

Compares Baseline vs Proposed (dataset_type=all). Optionally includes SummaC when
outputs_summac_baseline/{predictions,test_metrics}.json exist — same stratified split
as training (via run_summac_baseline.py).

HaluEval LLM-as-judge QA baseline belongs in the appendix — run qa_compare_baseline.py.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

BASELINE_CATEGORIES = {
    "Baseline (without signal reliability)": (
        "Learned detector — trained end-to-end on HaluEval (same encoder pipeline without "
        "signal-reliability branch; outputs P(hallucination))"
    ),
    "Proposed (with signal reliability)": (
        "Our method — learned detector with semantic + uncertainty cues and "
        "SignalReliabilityAnalyzer (outputs P(hallucination))"
    ),
    "SummaC (published consistency)": (
        "Published external baseline — SummaC NLI-based document–summary consistency "
        "(Laban et al.); mapped to P(hallucination) via val-split normalization "
        "(see run_summac_baseline.py)"
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
        "Baseline (without signal reliability)": {"color": "b", "fill_alpha": 0.10},
        "Proposed (with signal reliability)": {"color": "g", "fill_alpha": 0.20},
        "SummaC (published consistency)": {"color": "darkorange", "fill_alpha": 0.12},
    }


def plot_pr_curve_comparison(model_preds: Dict[str, dict], output_path: str):
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
    ax.set_title("Precision-Recall Comparison (All Tasks)", fontsize=14)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"PR curve comparison saved: {output_path}")


def plot_roc_curve_comparison(model_preds: Dict[str, dict], output_path: str):
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
    ax.set_title("ROC Comparison (All Tasks)", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
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
    model_preds: Dict[str, dict],
    all_metrics: Dict[str, dict],
    output_path: str,
    title: str = "Calibration Curve Comparison (All Tasks)",
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


def print_metrics_comparison(all_metrics: Dict[str, dict], all_preds: Dict[str, dict]):
    print("\n" + "=" * 60)
    print("Main Comparison — All Tasks")
    print("=" * 60)

    print("\nBaseline categories (for paper / appendix):")
    for name in all_preds:
        print(f"  • {name}: {BASELINE_CATEGORIES.get(name, '(custom)')}")

    colw = max(38, max(len(n) for n in all_preds) + 2) if all_preds else 38
    print(f"\n{'Model':<{colw}} {'AUROC':<10} {'AUPR(AP)':<10} {'ECE':<10}")
    print("-" * 60)

    for name, preds in all_preds.items():
        labels = np.array(preds["labels"])
        probs = np.array(preds["probs"])
        fpr, tpr, _ = roc_curve(labels, probs)
        model_auc = auc(fpr, tpr)
        model_ap = average_precision_score(labels, probs)
        model_ece = all_metrics.get(name, {}).get("metrics", {}).get("ece", float("nan"))
        ece_str = f"{model_ece:.4f}" if not np.isnan(model_ece) else "N/A"
        print(f"{name:<{colw}} {model_auc:<10.4f} {model_ap:<10.4f} {ece_str:<10}")

    print("=" * 60)


def _load_one(name: str, model_dir: str):
    pred_path = os.path.join(model_dir, "predictions.json")
    metrics_path = os.path.join(model_dir, "test_metrics.json")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"{name}: missing {pred_path}")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"{name}: missing {metrics_path}")
    return load_predictions(pred_path), load_metrics(metrics_path)


def main():
    parser = argparse.ArgumentParser(description="Main Baseline vs Proposed (+ optional SummaC).")
    parser.add_argument("--baseline_dir", default="./outputs_failure_aware")
    parser.add_argument("--proposed_dir", default="./outputs_failure_aware_semantic_signal_reliability")
    parser.add_argument(
        "--summac_dir",
        default="./outputs_summac_baseline",
        help="If predictions.json + test_metrics.json exist, include SummaC curve.",
    )
    parser.add_argument("--output_dir", default="./outputs_comparison")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    order = [
        "Baseline (without signal reliability)",
        "Proposed (with signal reliability)",
    ]
    model_dirs = {
        order[0]: args.baseline_dir,
        order[1]: args.proposed_dir,
    }

    summac_pred = os.path.join(args.summac_dir, "predictions.json")
    summac_met = os.path.join(args.summac_dir, "test_metrics.json")
    if os.path.isfile(summac_pred) and os.path.isfile(summac_met):
        order.append("SummaC (published consistency)")
        model_dirs["SummaC (published consistency)"] = args.summac_dir
        print(f"Including SummaC from {args.summac_dir}")
    else:
        print(
            f"No SummaC outputs at {args.summac_dir} — skipping. Run:\n"
            f"  python run_summac_baseline.py --output_dir {args.summac_dir}"
        )

    all_preds = {}
    all_metrics = {}
    ref_n = None

    for name in order:
        preds, metrics = _load_one(name, model_dirs[name])
        n = len(preds["labels"])
        if ref_n is None:
            ref_n = n
        elif n != ref_n:
            raise ValueError(
                f"Sample count mismatch for {name}: got {n}, expected {ref_n}. "
                "Regenerate SummaC with same --seed/--test_ratio/--val_ratio/--data_dir as training."
            )
        all_preds[name] = preds
        all_metrics[name] = metrics

    print(
        "Scheme 2: this plot compares local models on all tasks only.\n"
        "For QA-aligned HaluEval judge baseline vs same-split QA models, run:\n"
        "  python qa_compare_baseline.py"
    )
    print("Loaded models:")
    for name in all_preds:
        print(f"- {name}: {len(all_preds[name]['labels'])} samples")

    plot_pr_curve_comparison(all_preds, os.path.join(args.output_dir, "pr_curve_comparison.png"))
    plot_roc_curve_comparison(all_preds, os.path.join(args.output_dir, "roc_curve_comparison.png"))
    plot_calibration_curve_comparison(
        all_preds,
        all_metrics,
        os.path.join(args.output_dir, "calibration_comparison.png"),
    )
    print_metrics_comparison(all_metrics, all_preds)

    legend_path = os.path.join(args.output_dir, "baseline_legend.json")
    with open(legend_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "description": "Main comparison baseline categories",
                "models": {k: BASELINE_CATEGORIES.get(k, "") for k in all_preds},
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Baseline legend written to {legend_path}")

    print(f"\nMain comparison results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
