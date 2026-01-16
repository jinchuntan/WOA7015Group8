# src/eval_baseline.py
# Written by: Nigel Tan Jin Chun, 24054001
# Last Modified: 10/1/2026
# Purpose: Load a trained baseline (ResNet50 + BiLSTM) checkpoint, run inference on the VQA-RAD test split
#          using the same filtering/preprocessing as training, compute metrics, and save
#          metrics.json, predictions.csv, and kept_indices.json (for matching subsets across models).

from __future__ import annotations

import os

# Avoid common OpenMP / MKL runtime issues in some environments (e.g., Windows/Conda).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Keep CPU thread usage predictable (reduces variability / oversubscription).
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .baseline_model import BaselineVQAModel
from .data_vqarad import VQARadBaselineDataset, collate_baseline, load_vqarad_splits
from .metrics import eval_from_strings
from .text_utils import infer_qtype_from_gt, norm_ans


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Write a list of dict rows to CSV with a fixed column order."""
    path.parent.mkdir(parents=True, exist_ok=True)  # ensure output directory exists
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


@torch.no_grad()
def main():
    # ---- CLI arguments ----
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="outputs/baseline/baseline.pt")       # trained checkpoint path
    ap.add_argument("--out_dir", type=str, default="outputs/baseline_eval")           # output folder for eval artifacts
    args = ap.parse_args()

    # ---- Device + outputs ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load checkpoint ----
    ckpt = torch.load(args.ckpt, map_location="cpu")  # load on CPU first, then move model to GPU if needed
    vocab_q = ckpt["vocab_q"]                          # question vocab used during training
    ans2id = ckpt["ans2id"]                            # answer vocab mapping used during training
    id2ans = {i: a for a, i in ans2id.items()}         # inverse mapping for decoding predictions

    # Pull training-time settings to ensure test preprocessing matches training.
    train_args = ckpt.get("train_args", {})
    img_size = int(train_args.get("img_size", 224))
    max_q_len = int(train_args.get("max_q_len", 30))
    freeze_cnn = bool(int(train_args.get("freeze_cnn", 1)))  # stored as int-like flag in some runs

    # ---- Load dataset splits ----
    # We only need the test split; seed matters only if the dataset build creates a val split via splitting.
    _, _, test_split = load_vqarad_splits(seed=int(train_args.get("seed", 42)))

    # Build the test dataset using the same settings as training (including OOV filtering).
    test_ds = VQARadBaselineDataset(
        test_split,
        vocab_q=vocab_q,
        ans2id=ans2id,
        max_q_len=max_q_len,
        img_size=img_size,
        filter_oov=True,            # match training if training filtered OOV answers
        keep_orig_indices=True,     # keep indices so we can export the exact subset evaluated
    )

    # DataLoader with a custom collate function to stack tensors and keep text fields as lists.
    loader = DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_baseline,
    )

    # ---- Rebuild model and load weights ----
    model = BaselineVQAModel(
        n_answers=len(ans2id),
        vocab_size=len(vocab_q),
        freeze_cnn=freeze_cnn,
    ).to(device)
    
    # strict to ensure exact architecture match
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    # Cache yes/no answer IDs (used for a small improvement on closed-ended questions).
    yes_id = ans2id.get("yes", None)
    no_id = ans2id.get("no", None)

    # Storage for evaluation outputs across all batches.
    preds, gts, qtypes, questions, orig_idx = [], [], [], [], []

    # ---- Inference loop ----
    for batch in tqdm(loader, desc="predict"):
        # Move tensors to the selected device.
        images = batch["images"].to(device)
        q_ids = batch["q_ids"].to(device)
        q_lens = batch["q_lens"].to(device)

        # Forward pass: logits over answer vocabulary.
        logits = model(images, q_ids, q_lens)

        # Normalize GT answers and infer question types from GT (project heuristic).
        gt_txt = [norm_ans(x) for x in batch["answers"]]
        qt = [infer_qtype_from_gt(g) for g in gt_txt]

        # Convert logits -> predicted answer IDs.
        pred_ids = []
        for i in range(logits.size(0)):
            # If closed-ended and yes/no exist in vocab, only compare those two logits.
            # This prevents the model from picking unrelated tokens for yes/no questions.
            if qt[i] == "closed" and (yes_id is not None) and (no_id is not None):
                sub = logits[i, torch.tensor([no_id, yes_id], device=logits.device)]
                pick = torch.argmax(sub).item()
                pred_ids.append(yes_id if pick == 1 else no_id)
            else:
                # For open-ended (or if yes/no are missing), take argmax over the full vocabulary.
                pred_ids.append(int(torch.argmax(logits[i]).item()))

        # Decode predicted IDs to text and normalize for fair exact-match evaluation.
        pred_txt = [norm_ans(id2ans[i]) for i in pred_ids]

        # Accumulate per-example fields for metrics + CSV export.
        preds.extend(pred_txt)
        gts.extend(gt_txt)
        qtypes.extend(qt)
        questions.extend(batch["questions"])
        orig_idx.extend(batch["orig_index"])

    # ---- Metrics + outputs ----
    metrics = eval_from_strings(gts, preds, qtypes)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Save which indices from the ORIGINAL test split were kept after filtering.
    # This allows BLIP evaluation on the identical subset for apples-to-apples comparison.
    (out_dir / "kept_indices.json").write_text(json.dumps(orig_idx, indent=2), encoding="utf-8")

    # Build a per-example predictions table for debugging and reporting.
    rows = []
    for q, qt, gt, pr, oi in zip(questions, qtypes, gts, preds, orig_idx):
        rows.append(
            {
                "orig_index": oi,
                "question": q,
                "qtype": qt,
                "gt": gt,
                "pred": pr,
                "correct": int(gt == pr),
            }
        )

    write_csv(out_dir / "predictions.csv", rows, ["orig_index", "question", "qtype", "gt", "pred", "correct"])

    # Print summary to terminal for quick inspection.
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved: {out_dir/'metrics.json'} and {out_dir/'predictions.csv'}")
    print(f"Saved: {out_dir/'kept_indices.json'}")


if __name__ == "__main__":
    main()
