# src/eval_blip.py
# Written by: Nigel Tan Jin Chun, 24054001
# Last Modified: 10/1/2026
# Purpose: Evaluate BLIP on the VQA-RAD test split (optionally using a fixed subset from baseline kept_indices.json
#          for fair comparison), compute metrics, and save metrics.json + predictions.csv. Supports optional PEFT/LoRA
#          adapter loading for adapted BLIP checkpoints.

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
from typing import List, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import BlipForQuestionAnswering, BlipProcessor

from .metrics import eval_from_strings
from .text_utils import infer_qtype_from_gt, norm_ans


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Write a list of dict rows to CSV with a fixed column order."""

    # ensure output folder exists
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


@torch.inference_mode()
def pick_yes_no_by_loss(model, processor, img, question: str, device: str) -> str:
    """
    Closed-ended handling: choose between 'yes' and 'no' by comparing the loss for each forced label.
    This is typically more reliable than free-form generation for yes/no questions.
    """
    # Build model inputs for the image-question pair.
    inputs = processor(img, question, return_tensors="pt").to(device)

    # Tokenize the candidate labels ("yes" and "no") as supervised targets.
    yes_ids = processor.tokenizer("yes", return_tensors="pt").input_ids.to(device)
    no_ids = processor.tokenizer("no", return_tensors="pt").input_ids.to(device)

    # Compute negative log-likelihood via loss; lower loss means the label is more likely.
    loss_yes = model(**inputs, labels=yes_ids).loss.item()
    loss_no = model(**inputs, labels=no_ids).loss.item()

    return "yes" if loss_yes < loss_no else "no"


def postprocess_open(pred: str) -> str:
    """Light cleanup for open-ended generations before normalization."""

    # robust to None/empty outputs
    p = (pred or "").strip().lower()

    # Heuristic: keep only the first segment before punctuation/newline to avoid extra text.
    for sep in [".", ",", ";", ":", "\n"]:
        if sep in p:
            p = p.split(sep)[0].strip()

    # apply project-wide normalization rules
    return norm_ans(p)


@torch.inference_mode()
def run_blip(
    model,
    processor,
    split,
    device: str,
    subset_idx_list: Optional[List[int]] = None,
    max_new_tokens: int = 10,
    num_beams: int = 3,
    limit_n: int = 0,
):
    """
    Run BLIP evaluation over a HuggingFace dataset split.

    Returns aligned lists for:
    - preds: normalized predictions
    - gts: normalized ground truths
    - qtypes: inferred question types ("closed"/"open")
    - questions: raw question strings
    - orig_idx: original indices in the full test split (especially important when using a subset)
    """
    preds: List[str] = []
    gts: List[str] = []
    qtypes: List[str] = []
    questions: List[str] = []
    orig_idx: List[int] = []

    # Optionally limit the number of evaluated examples for quick test runs.
    n = len(split)
    if limit_n and limit_n > 0:
        n = min(n, limit_n)

    for i in tqdm(range(n), desc="BLIP eval"):
        ex = split[i]

        # Extract raw fields from the dataset example.
        # PIL image (HuggingFace image feature)
        img = ex["image"]

        # question string      
        q = ex["question"]         

        # normalize ground truth answer for fair exact-match scoring
        gt = norm_ans(ex["answer"])
        qtype = infer_qtype_from_gt(gt)

        if qtype == "closed":

            # Closed-ended: force "yes" and "no" and pick the lower-loss one.
            pred = pick_yes_no_by_loss(model, processor, img, q, device)
        else:
            # Open-ended: generate a short textual answer using beam search.
            inputs = processor(img, q, return_tensors="pt").to(device)
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,  # keep answers short/consistent
                num_beams=num_beams,            # more stable decoding than greedy
                do_sample=False,                # deterministic evaluation
                early_stopping=True,
            )
            pred = processor.decode(out[0], skip_special_tokens=True)
            pred = postprocess_open(pred)

        # Store outputs for later metric computation and CSV export.
        preds.append(pred)
        gts.append(gt)
        qtypes.append(qtype)
        questions.append(q)

        # Map back to the original test index if we evaluated a selected subset.
        orig_idx.append(int(subset_idx_list[i]) if subset_idx_list is not None else int(i))

    return preds, gts, qtypes, questions, orig_idx


def main():
    # ---- CLI arguments ----
    ap = argparse.ArgumentParser()

    # base BLIP checkpoint
    ap.add_argument("--model_name", type=str, default="Salesforce/blip-vqa-base")

    # output directory
    ap.add_argument("--out_dir", type=str, default="outputs/blip_eval")           

    # open-ended gen length
    ap.add_argument("--max_new_tokens", type=int, default=10)                     
    
    # beam width
    ap.add_argument("--num_beams", type=int, default=3)

     # optional cap on examples                          
    ap.add_argument("--limit_n", type=int, default=0)                            

    # Fairness: evaluate on the same filtered subset as the baseline (kept_indices.json).
    ap.add_argument("--subset_indices", type=str, default="", help="Path to kept_indices.json from baseline_eval")

    # Optional: load a PEFT adapter (e.g., LoRA) onto the base BLIP model.
    ap.add_argument("--lora_path", type=str, default="", help="Optional PEFT adapter folder")

    args = ap.parse_args()

    # ---- Device + outputs ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load dataset ----
    ds = load_dataset("flaviagiammarino/vqa-rad")
    test_split = ds["test"]

    # Optionally restrict the test set to a known subset for apples-to-apples comparison.
    subset_idx_list = None
    if args.subset_indices:
        subset_idx_list = json.loads(Path(args.subset_indices).read_text(encoding="utf-8"))
        test_split = test_split.select(subset_idx_list)

    # ---- Load BLIP model + processor ----
    processor = BlipProcessor.from_pretrained(args.model_name)
    model = BlipForQuestionAnswering.from_pretrained(args.model_name).to(device)

    # Optionally attach a PEFT adapter (LoRA) if provided.
    if args.lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_path).to(device)

    model.eval()

    # ---- Run evaluation ----
    preds, gts, qtypes, questions, orig_idx = run_blip(
        model=model,
        processor=processor,
        split=test_split,
        device=device,
        subset_idx_list=subset_idx_list,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        limit_n=args.limit_n,
    )

    # ---- Metrics + outputs ----
    metrics = eval_from_strings(gts, preds, qtypes)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Build a per-example table for inspection/debugging.
    rows = []
    for oi, q, qt, gt, pr in zip(orig_idx, questions, qtypes, gts, preds):
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


if __name__ == "__main__":
    main()
