# src/metrics.py
# Written by: Nigel Tan Jin Chun, 24054001
# Last Modified: 10/1/2026
# Purpose: Compute evaluation metrics for VQA predictions given ground-truth answers, predicted answers,
#          and question types ("closed"/"open"). Outputs overall exact match, closed-ended yes/no metrics
#          (accuracy, precision/recall/F1 for "yes", confusion matrix), and open-ended similarity metrics
#          (exact match, token-F1, ROUGE-L F1).

from __future__ import annotations

from typing import Dict, List, Tuple

from .text_utils import norm_ans, simple_tokenize  # shared normalization + lightweight tokenization


def _token_f1(gt: str, pr: str) -> float:
    """Token-level F1 using multiset (count-based) overlap between GT and prediction tokens."""
    gt_t = simple_tokenize(gt)
    pr_t = simple_tokenize(pr)

    # Handle empty strings explicitly.
    if not gt_t and not pr_t:
        return 1.0
    if not gt_t or not pr_t:
        return 0.0

    # Treat token lists as multisets and compute overlap by counts.
    from collections import Counter
    c_gt = Counter(gt_t)
    c_pr = Counter(pr_t)

    overlap = 0
    for k in c_gt.keys():
        overlap += min(c_gt[k], c_pr.get(k, 0))

    # Precision/recall in token space.
    prec = overlap / max(1, sum(c_pr.values()))
    rec = overlap / max(1, sum(c_gt.values()))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _lcs_len(a: List[str], b: List[str]) -> int:
    """Length of Longest Common Subsequence (LCS) between two token sequences (DP; fine for short strings)."""
    n, m = len(a), len(b)

    # dp[i][j] = LCS length between a[:i] and b[:j]
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


def _rouge_l_f1(gt: str, pr: str) -> float:
    """ROUGE-L F1 computed from token-level LCS precision and recall."""
    gt_t = simple_tokenize(gt)
    pr_t = simple_tokenize(pr)

    # Handle empty strings explicitly.
    if not gt_t and not pr_t:
        return 1.0
    if not gt_t or not pr_t:
        return 0.0

    # LCS-based precision/recall.
    lcs = _lcs_len(gt_t, pr_t)
    prec = lcs / max(1, len(pr_t))
    rec = lcs / max(1, len(gt_t))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def eval_from_strings(gts: List[str], preds: List[str], qtypes: List[str]) -> Dict:
    """
    Compute overall + per-type metrics.

    Inputs:
      - gts: ground-truth answer strings
      - preds: predicted answer strings
      - qtypes: list of "closed"/"open" (same length as gts/preds)

    Notes:
      - All strings are normalized via norm_ans() before scoring.
      - Closed-ended scoring treats "yes" as the positive class and builds a 2x2 confusion matrix.
    """

    # safety check for aligned lists
    assert len(gts) == len(preds) == len(qtypes)

    # Normalize all answers to ensure consistent evaluation (case/punct/spacing rules).
    gts_n = [norm_ans(x) for x in gts]
    prs_n = [norm_ans(x) for x in preds]

    n_total = len(gts_n)

    # Indices for each question type.
    closed_idx = [i for i, t in enumerate(qtypes) if t == "closed"]
    open_idx = [i for i, t in enumerate(qtypes) if t == "open"]

    # ---- Overall exact match ----
    overall_em = sum(int(gts_n[i] == prs_n[i]) for i in range(n_total)) / max(1, n_total)

    # ---- Closed-ended metrics (yes/no) ----
    # Confusion matrix uses "yes" as positive:
    # [[TN, FP],
    #  [FN, TP]]
    tp = fp = tn = fn = 0
    for i in closed_idx:
        gt = gts_n[i]
        pr = prs_n[i]

        gt_yes = (gt == "yes")
        pr_yes = (pr == "yes")

        if gt_yes and pr_yes:
            tp += 1
        elif (not gt_yes) and pr_yes:
            fp += 1
        elif gt_yes and (not pr_yes):
            fn += 1
        else:
            tn += 1

    closed_acc = (tp + tn) / max(1, len(closed_idx))

    # precision for the "yes" class
    prec_yes = tp / max(1, tp + fp)

    # recall for the "yes" class
    rec_yes = tp / max(1, tp + fn)
    f1_yes = 0.0 if (prec_yes + rec_yes) == 0 else (2 * prec_yes * rec_yes / (prec_yes + rec_yes))

    # ---- Open-ended metrics ----
    open_em = sum(int(gts_n[i] == prs_n[i]) for i in open_idx) / max(1, len(open_idx))
    open_token_f1 = sum(_token_f1(gts_n[i], prs_n[i]) for i in open_idx) / max(1, len(open_idx))
    open_rougeL_f1 = sum(_rouge_l_f1(gts_n[i], prs_n[i]) for i in open_idx) / max(1, len(open_idx))

    # Return a JSON-serializable dict for saving to metrics.json.
    return {
        "closed_accuracy": float(closed_acc),
        "closed_precision_yes": float(prec_yes),
        "closed_recall_yes": float(rec_yes),
        "closed_f1_yes": float(f1_yes),
        "closed_confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "open_exact_match": float(open_em),
        "open_token_f1": float(open_token_f1),
        "open_rougeL_f1": float(open_rougeL_f1),
        "overall_exact_match": float(overall_em),
        "n_total": int(n_total),
        "n_closed": int(len(closed_idx)),
        "n_open": int(len(open_idx)),
    }
