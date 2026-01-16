# src/train_baseline.py
# Written by: Nigel Tan Jin Chun, 24054001
# Last Modified: 10/1/2026
# Purpose: Train the baseline VQA model (ResNet50 image encoder + BiLSTM question encoder) on VQA-RAD by
#          building a top-K answer vocabulary from training data, filtering out OOV answers, training with
#          cross-entropy over answer classes, evaluating on val/test each epoch, and saving the best checkpoint
#          (baseline.pt) plus a running training history (train_history.json).

from __future__ import annotations

import os

# Avoid common OpenMP / MKL runtime issues in some environments (e.g., Windows/Conda).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Keep CPU thread usage predictable (reduces variability / oversubscription).
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .baseline_model import BaselineVQAModel
from .data_vqarad import VQARadBaselineDataset, collate_baseline, load_vqarad_splits
from .metrics import eval_from_strings
from .text_utils import build_question_vocab, infer_qtype_from_gt, norm_ans


def set_seed(seed: int):
    """Set random seeds for reproducibility across Python, NumPy, and Torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict_strings(model, loader, id2ans: dict[int, str], ans2id: dict[str, int], device: str):
    """
    Run inference and return GT/pred strings + qtypes for metric computation.

    Key accuracy improvement:
      - For CLOSED questions, restrict prediction to only "yes" vs "no" (if present in the answer vocab).
      - For OPEN questions, use argmax over the full answer vocabulary.
    """
    model.eval()

    # Cache IDs for yes/no if they exist in the answer vocabulary.
    yes_id = ans2id.get("yes", None)
    no_id = ans2id.get("no", None)

    gts, preds, qtypes = [], [], []

    for batch in loader:
        # Move tensor inputs to the selected device.
        images = batch["images"].to(device)
        q_ids = batch["q_ids"].to(device)
        q_lens = batch["q_lens"].to(device)

        # Forward pass: logits over answer classes [B, K].
        logits = model(images, q_ids, q_lens)

        # Normalize GT text and infer question type from GT (project heuristic).
        gt_txt = [norm_ans(x) for x in batch["answers"]]
        qt = [infer_qtype_from_gt(g) for g in gt_txt]

        # Convert logits to predicted answer IDs with closed/open handling.
        pred_ids = []
        for i in range(logits.size(0)):
            if qt[i] == "closed" and (yes_id is not None) and (no_id is not None):
                # Only compare logits for {no, yes} to avoid selecting unrelated classes.
                sub = logits[i, torch.tensor([no_id, yes_id], device=logits.device)]
                pick = torch.argmax(sub).item()
                pred_ids.append(yes_id if pick == 1 else no_id)
            else:
                # Open-ended: choose the most probable class in the full answer vocab.
                pred_ids.append(int(torch.argmax(logits[i]).item()))

        # Decode IDs back to normalized text for fair exact-match scoring.
        pred_txt = [norm_ans(id2ans[i]) for i in pred_ids]

        # Accumulate results across batches.
        preds.extend(pred_txt)
        gts.extend(gt_txt)
        qtypes.extend(qt)

    return gts, preds, qtypes


@torch.no_grad()
def evaluate(model, loader, id2ans, ans2id, device: str):
    """Compute eval metrics on a loader using string-based predictions."""
    gts, preds, qtypes = predict_strings(model, loader, id2ans, ans2id, device)
    return eval_from_strings(gts, preds, qtypes)


def main():
    # ---- CLI arguments ----
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=100)         # size of answer vocabulary (most frequent answers)
    ap.add_argument("--epochs", type=int, default=9)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img_size", type=int, default=224)     # image resize for CNN input
    ap.add_argument("--max_q_len", type=int, default=30)     # max token length for question encoding
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--freeze_cnn", type=int, default=1, help="1=frozen ResNet, 0=train ResNet (slow on CPU)")
    ap.add_argument("--out_dir", type=str, default="outputs/baseline")
    args = ap.parse_args()

    # ---- Setup ----
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset splits (will create a val split if the dataset does not provide one).
    train_split, val_split, test_split = load_vqarad_splits(seed=args.seed)

    # ---- Build answer vocabulary (top-K answers from TRAIN) ----
    # The baseline is a classification model, so it can only predict answers in a fixed vocab.
    train_answers = [norm_ans(x["answer"]) for x in train_split]
    cnt = Counter(train_answers)
    top_answers = [a for a, _ in cnt.most_common(args.topk)]
    ans2id = {a: i for i, a in enumerate(top_answers)}
    id2ans = {i: a for a, i in ans2id.items()}

    # Quick sanity prints to interpret loss magnitude (cross-entropy ~ ln(K) at random guessing).
    print(f"Answer classes (K) = {len(ans2id)}  |  ln(K) = {math.log(max(2, len(ans2id))):.3f}")
    print(f'Contains "yes": {"yes" in ans2id} | Contains "no": {"no" in ans2id}')

    # ---- Build question vocabulary (from TRAIN questions) ----
    train_questions = [x["question"] for x in train_split]
    vocab_q = build_question_vocab(train_questions, min_freq=1, max_size=20000)

    # ---- Build datasets (filter out OOV answers not in top-K) ----
    train_ds = VQARadBaselineDataset(train_split, vocab_q, ans2id, args.max_q_len, args.img_size, filter_oov=True)
    val_ds = VQARadBaselineDataset(val_split, vocab_q, ans2id, args.max_q_len, args.img_size, filter_oov=True)
    test_ds = VQARadBaselineDataset(test_split, vocab_q, ans2id, args.max_q_len, args.img_size, filter_oov=True)

    # Report how many examples remain after filtering.
    print(f"Train: {len(train_split)} -> kept {len(train_ds)}")
    print(f"Val:   {len(val_split)} -> kept {len(val_ds)}")
    print(f"Test:  {len(test_split)} -> kept {len(test_ds)}")

    # ---- DataLoaders ----
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_baseline,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_baseline,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_baseline,
    )

    # ---- Model ----
    model = BaselineVQAModel(
        n_answers=len(ans2id),
        vocab_size=len(vocab_q),
        freeze_cnn=bool(args.freeze_cnn),
    ).to(device)

    # Only optimize parameters that require gradients (important if CNN is frozen).
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()  # classification over answer classes

    # Track per-epoch records and keep the best checkpoint by validation overall exact match.
    history = []
    best_val = -1.0

    # ---- Training loop ----
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for batch in tqdm(train_loader, desc=f"train epoch {epoch}/{args.epochs}"):
            # Move tensors to device.
            images = batch["images"].to(device)
            q_ids = batch["q_ids"].to(device)
            q_lens = batch["q_lens"].to(device)

            # Convert GT answers (strings) into answer IDs in ans2id.
            gt_txt = [norm_ans(x) for x in batch["answers"]]
            gt_ids = torch.tensor([ans2id[x] for x in gt_txt], dtype=torch.long, device=device)

            # Forward -> loss -> backward -> update.
            logits = model(images, q_ids, q_lens)
            loss = loss_fn(logits, gt_ids)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(float(loss.item()))

        # ---- Evaluate after each epoch ----
        val_metrics = evaluate(model, val_loader, id2ans, ans2id, device)
        test_metrics = evaluate(model, test_loader, id2ans, ans2id, device)

        # Record a compact summary for plotting/reporting.
        rec = {
            "epoch": epoch,
            "train_loss": float(sum(losses) / max(1, len(losses))),
            "val_overall_exact_match": float(val_metrics["overall_exact_match"]),
            "val_closed_accuracy": float(val_metrics["closed_accuracy"]),
            "val_open_exact_match": float(val_metrics["open_exact_match"]),
            "test_overall_exact_match": float(test_metrics["overall_exact_match"]),
            "test_closed_accuracy": float(test_metrics["closed_accuracy"]),
            "test_open_exact_match": float(test_metrics["open_exact_match"]),
        }
        history.append(rec)

        # ---- Save best checkpoint by validation overall EM ----
        if rec["val_overall_exact_match"] > best_val:
            best_val = rec["val_overall_exact_match"]
            ckpt = {
                "model_state": model.state_dict(),
                "vocab_q": vocab_q,
                "ans2id": ans2id,
                "train_args": vars(args),  # store training args so eval can match preprocessing
            }
            torch.save(ckpt, out_dir / "baseline.pt")

        # Persist training history each epoch so progress is not lost if the run stops.
        (out_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(rec, indent=2))

    # Final paths for convenience.
    print(f"\nSaved checkpoint: {out_dir/'baseline.pt'}")
    print(f"Saved training history: {out_dir/'train_history.json'}")


if __name__ == "__main__":
    main()
