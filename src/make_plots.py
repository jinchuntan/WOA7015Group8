# src/make_plots.py
# Written by: Nigel Tan Jin Chun, 24054001
# Last Modified: 10/1/2026
# Purpose: Generate training and evaluation plots for the baseline and BLIP experiments by loading
#          saved JSON/CSV artifacts (train_history.json, metrics.json, predictions.csv), computing
#          additional open-ended text similarity scores, and exporting PNG figures for reporting.

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# lightweight tokenizer used for token-level metrics
from .text_utils import simple_tokenize


def token_f1(gt: str, pr: str) -> float:
    """Compute token-level F1 between ground-truth and prediction (bag-of-words overlap)."""
    gt_t = simple_tokenize(gt)
    pr_t = simple_tokenize(pr)

    # Handle empty cases explicitly.
    if not gt_t and not pr_t:
        return 1.0
    if not gt_t or not pr_t:
        return 0.0

    # Count token overlaps (multiset overlap).
    from collections import Counter
    c_gt = Counter(gt_t)
    c_pr = Counter(pr_t)
    overlap = sum(min(c_gt[k], c_pr.get(k, 0)) for k in c_gt.keys())

    # Precision/recall over tokens.
    prec = overlap / max(1, sum(c_pr.values()))
    rec = overlap / max(1, sum(c_gt.values()))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def lcs_len(a: List[str], b: List[str]) -> int:
    """Compute length of the Longest Common Subsequence (LCS) between two token sequences."""
    n, m = len(a), len(b)
    
    # DP table for LCS; dp[i][j] = LCS length for a[:i] and b[:j].
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        ai = a[i - 1]
        row = dp[i]
        prev = dp[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                row[j] = prev[j - 1] + 1
            else:
                row[j] = max(prev[j], row[j - 1])
    return dp[n][m]


def rouge_l_f1(gt: str, pr: str) -> float:
    """Compute ROUGE-L F1 using token-level LCS."""
    gt_t = simple_tokenize(gt)
    pr_t = simple_tokenize(pr)

    # Handle empty cases explicitly.
    if not gt_t and not pr_t:
        return 1.0
    if not gt_t or not pr_t:
        return 0.0

    # LCS-based precision/recall.
    lcs = lcs_len(gt_t, pr_t)
    prec = lcs / max(1, len(pr_t))
    rec = lcs / max(1, len(gt_t))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def safe_load_json(path: Path) -> Optional[dict]:
    """Load JSON if it exists; otherwise print a skip message and return None."""
    if not path.exists():
        print(f"[skip] missing: {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def safe_load_csv(path: Path) -> Optional[pd.DataFrame]:
    """Load CSV if it exists; otherwise print a skip message and return None."""
    if not path.exists():
        print(f"[skip] missing: {path}")
        return None
    return pd.read_csv(path)


def save_line_plot(
    x: List[int],
    y: List[float],
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
):
    """Save a simple single-line plot to a PNG file."""
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_multi_line_plot(
    x: List[int],
    ys: Dict[str, List[float]],
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
):
    """Save a multi-line plot (several named series) to a PNG file."""
    plt.figure()
    for name, y in ys.items():
        if y is None:
            continue
        plt.plot(x, y, marker="o", label=name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_confusion_matrix(cm: List[List[int]], title: str, out_path: Path):
    """Save a 2x2 confusion matrix image for closed-ended (yes/no) questions."""
    cm_np = np.array(cm, dtype=int)
    plt.figure()
    plt.imshow(cm_np, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.xticks([0, 1], ["Pred No", "Pred Yes"])
    plt.yticks([0, 1], ["GT No", "GT Yes"])

    # Annotate each cell with its count.
    for i in range(cm_np.shape[0]):
        for j in range(cm_np.shape[1]):
            plt.text(j, i, str(cm_np[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_compare_bar(metrics_a: dict, metrics_b: dict, name_a: str, name_b: str, out_path: Path):
    """Save a side-by-side bar chart comparing key metrics between two models."""
    keys = [
        ("closed_accuracy", "Closed Acc"),
        ("closed_f1_yes", "Closed F1 (Yes)"),
        ("open_exact_match", "Open EM"),
        ("open_token_f1", "Open Token-F1"),
        ("open_rougeL_f1", "Open ROUGE-L"),
        ("overall_exact_match", "Overall EM"),
    ]
    vals_a = [float(metrics_a[k]) for k, _ in keys]
    vals_b = [float(metrics_b[k]) for k, _ in keys]
    labels = [lab for _, lab in keys]

    x = np.arange(len(labels))
    w = 0.35

    plt.figure(figsize=(10, 4))
    plt.bar(x - w / 2, vals_a, width=w, label=name_a)
    plt.bar(x + w / 2, vals_b, width=w, label=name_b)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylim(0, 1.0)
    plt.title("Baseline vs BLIP (Key Metrics)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def add_open_metrics_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add extra per-row analysis columns:
    - q_len: question length in tokens
    - token_f1 and rougeL_f1: soft text similarity scores for open-ended analysis
    """
    df = df.copy()
    df["q_len"] = df["question"].astype(str).apply(lambda s: len(simple_tokenize(s)))
    df["token_f1"] = df.apply(lambda r: token_f1(str(r["gt"]), str(r["pred"])), axis=1)
    df["rougeL_f1"] = df.apply(lambda r: rouge_l_f1(str(r["gt"]), str(r["pred"])), axis=1)
    return df


def save_open_histograms(df_base: pd.DataFrame, df_blip: pd.DataFrame, out_dir: Path, prefix: str):
    """Save histograms comparing Baseline vs BLIP distributions on open-ended similarity metrics."""
    # Filter to open-ended only.
    b = df_base[df_base["qtype"] == "open"]
    p = df_blip[df_blip["qtype"] == "open"]
    if len(b) == 0 or len(p) == 0:
        print("[skip] open histograms: missing open rows")
        return

    # Token-F1 distribution.
    plt.figure()
    plt.hist(b["token_f1"], bins=15, alpha=0.6, label="Baseline")
    plt.hist(p["token_f1"], bins=15, alpha=0.6, label="BLIP")
    plt.title("Open Questions: Token-F1 Distribution")
    plt.xlabel("Token-F1")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_open_token_f1_hist.png", dpi=150)
    plt.close()

    # ROUGE-L distribution.
    plt.figure()
    plt.hist(b["rougeL_f1"], bins=15, alpha=0.6, label="Baseline")
    plt.hist(p["rougeL_f1"], bins=15, alpha=0.6, label="BLIP")
    plt.title("Open Questions: ROUGE-L F1 Distribution")
    plt.xlabel("ROUGE-L F1")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_open_rougeL_hist.png", dpi=150)
    plt.close()


def save_top_wrong_answers(df: pd.DataFrame, title: str, out_path: Path, top_n: int = 15):
    """Plot the most frequent GT answers among incorrect predictions (quick error profiling)."""
    wrong = df[df["correct"] == 0]
    if len(wrong) == 0:
        print(f"[skip] {title}: no wrong rows")
        return

    counts = wrong["gt"].value_counts().head(top_n)
    plt.figure(figsize=(10, 4))
    plt.bar(counts.index.astype(str), counts.values)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_question_length_boxplot(df: pd.DataFrame, title: str, out_path: Path):
    """Boxplot comparing question token length for correct vs wrong predictions."""
    if "q_len" not in df.columns:
        return

    correct = df[df["correct"] == 1]["q_len"].values
    wrong = df[df["correct"] == 0]["q_len"].values
    if len(correct) == 0 or len(wrong) == 0:
        print(f"[skip] {title}: not enough rows")
        return

    plt.figure()
    plt.boxplot([correct, wrong], labels=["Correct", "Wrong"])
    plt.title(title)
    plt.ylabel("Question length (tokens)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_yes_bias_bar(df_base: pd.DataFrame, df_blip: pd.DataFrame, out_path: Path):
    """Compare the rate of predicting 'yes' on closed-ended questions (simple bias check)."""
    def yes_rate(df: pd.DataFrame) -> float:
        closed = df[df["qtype"] == "closed"]
        if len(closed) == 0:
            return 0.0
        return float((closed["pred"] == "yes").mean())

    r_base = yes_rate(df_base)
    r_blip = yes_rate(df_blip)

    plt.figure()
    plt.bar(["Baseline", "BLIP"], [r_base, r_blip])
    plt.ylim(0, 1.0)
    plt.title("Closed Questions: Predicted 'Yes' Rate (Bias Check)")
    plt.ylabel("Rate")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    # ---- CLI arguments ----
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_history", type=str, default="outputs/baseline/train_history.json")  # baseline logs
    ap.add_argument("--baseline_eval_dir", type=str, default="outputs/baseline_eval")            # baseline eval folder
    ap.add_argument("--blip_eval_dir", type=str, default="outputs/blip_eval")                    # BLIP eval folder
    ap.add_argument("--out_dir", type=str, default="outputs/plots")                              # plot output folder
    args = ap.parse_args()

    # Timestamp prefix so each run produces uniquely named plots.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # 1) Training curves (baseline)
    # -------------------------
    hist_path = Path(args.train_history)
    hist = safe_load_json(hist_path)
    if hist:
        epochs = [h["epoch"] for h in hist]
        train_loss = [h.get("train_loss") for h in hist]

        # Training loss curve.
        save_line_plot(
            epochs, train_loss,
            "Epoch", "Train Loss",
            "Baseline Training Loss",
            out_dir / f"{ts}_train_loss.png"
        )

        # Validation overall exact match, if present in the history file.
        if "val_overall_exact_match" in hist[0]:
            save_line_plot(
                epochs,
                [h["val_overall_exact_match"] for h in hist],
                "Epoch", "Val Overall EM",
                "Baseline Validation Overall Exact Match",
                out_dir / f"{ts}_val_overall_em.png"
            )

        # Validation closed vs open curves, if present (or partially present).
        val_closed = [h.get("val_closed_accuracy") for h in hist]
        val_open = [h.get("val_open_exact_match") for h in hist]
        save_multi_line_plot(
            epochs,
            {"Val Closed Acc": val_closed, "Val Open EM": val_open},
            "Epoch", "Score",
            "Baseline Validation: Closed vs Open",
            out_dir / f"{ts}_val_closed_vs_open.png"
        )

        # Test overall exact match curve, if present in the history file.
        if "test_overall_exact_match" in hist[0]:
            save_line_plot(
                epochs,
                [h["test_overall_exact_match"] for h in hist],
                "Epoch", "Test Overall EM",
                "Baseline Test Overall Exact Match",
                out_dir / f"{ts}_test_overall_em.png"
            )

        # Test closed vs open curves, if present.
        if "test_closed_accuracy" in hist[0] and "test_open_exact_match" in hist[0]:
            save_multi_line_plot(
                epochs,
                {"Test Closed Acc": [h["test_closed_accuracy"] for h in hist],
                 "Test Open EM": [h["test_open_exact_match"] for h in hist]},
                "Epoch", "Score",
                "Baseline Test: Closed vs Open",
                out_dir / f"{ts}_test_closed_vs_open.png"
            )

    # -------------------------
    # 2) Eval metrics + confusion matrices + comparisons
    # -------------------------
    base_eval_dir = Path(args.baseline_eval_dir)
    blip_eval_dir = Path(args.blip_eval_dir)

    base_metrics = safe_load_json(base_eval_dir / "metrics.json")
    blip_metrics = safe_load_json(blip_eval_dir / "metrics.json")

    # Closed-ended confusion matrices (2x2) if included in metrics.json.
    if base_metrics and "closed_confusion_matrix" in base_metrics:
        save_confusion_matrix(
            base_metrics["closed_confusion_matrix"],
            "Baseline Closed Confusion Matrix",
            out_dir / f"{ts}_baseline_closed_confusion.png"
        )

    if blip_metrics and "closed_confusion_matrix" in blip_metrics:
        save_confusion_matrix(
            blip_metrics["closed_confusion_matrix"],
            "BLIP Closed Confusion Matrix",
            out_dir / f"{ts}_blip_closed_confusion.png"
        )

    # Side-by-side key metric comparison bar chart.
    if base_metrics and blip_metrics:
        save_compare_bar(
            base_metrics, blip_metrics,
            "Baseline", "BLIP",
            out_dir / f"{ts}_compare_baseline_vs_blip.png"
        )

    # -------------------------
    # 3) Prediction-level analysis (from predictions.csv)
    # -------------------------
    df_base = safe_load_csv(base_eval_dir / "predictions.csv")
    df_blip = safe_load_csv(blip_eval_dir / "predictions.csv")

    # Baseline prediction-level plots.
    if df_base is not None:
        df_base = add_open_metrics_columns(df_base)
        save_top_wrong_answers(
            df_base,
            "Baseline: Most Frequent Wrong Ground-Truth Answers",
            out_dir / f"{ts}_baseline_top_wrong_gt.png"
        )
        save_question_length_boxplot(
            df_base,
            "Baseline: Question Length vs Correctness",
            out_dir / f"{ts}_baseline_q_len_boxplot.png"
        )

    # BLIP prediction-level plots.
    if df_blip is not None:
        df_blip = add_open_metrics_columns(df_blip)
        save_top_wrong_answers(
            df_blip,
            "BLIP: Most Frequent Wrong Ground-Truth Answers",
            out_dir / f"{ts}_blip_top_wrong_gt.png"
        )
        save_question_length_boxplot(
            df_blip,
            "BLIP: Question Length vs Correctness",
            out_dir / f"{ts}_blip_q_len_boxplot.png"
        )

    # Comparative prediction-level plots (need both).
    if df_base is not None and df_blip is not None:
        save_open_histograms(df_base, df_blip, out_dir, ts)
        save_yes_bias_bar(df_base, df_blip, out_dir / f"{ts}_yes_bias_bar.png")

    print(f"\nSaved plots to: {out_dir}")


if __name__ == "__main__":
    main()
