# src/finetune_blip_lora.py
# Written by: Nigel Tan Jin Chun, 24054001
# Last Modified: 10/1/2026
# Purpose: Fine-tune BLIP for VQA-RAD using PEFT LoRA adapters (lightweight training),
#          evaluate after each epoch, and save the adapter weights + processor + training log
#          into an output directory for later loading during evaluation.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BlipForQuestionAnswering, BlipProcessor

from .metrics import eval_from_strings
from .text_utils import constrain_closed_prediction, infer_qtype_from_gt, norm_ans


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across torch + numpy (and CUDA if available)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class VQARadBlipDataset(torch.utils.data.Dataset):
    """
    Minimal dataset wrapper for BLIP finetuning:
    returns raw PIL images, question strings, and normalized answer strings.
    """
    def __init__(self, split):
        self.split = split

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        ex = self.split[idx]
        return {
            "image": ex["image"],
            "question": ex["question"],
            "answer": norm_ans(ex["answer"]),  # normalize answers so training/eval use consistent text
        }


def make_collate(processor: BlipProcessor, device: str):
    """
    Create a collate_fn that:
    - runs BLIP processor on images + questions (handles padding)
    - tokenizes answers as labels
    - masks label padding tokens with -100 so they do not contribute to loss
    """
    pad_id = processor.tokenizer.pad_token_id

    def collate(batch: List[Dict[str, object]]):
        # Collect raw fields from the dataset items.
        images = [b["image"] for b in batch]
        questions = [b["question"] for b in batch]
        answers = [b["answer"] for b in batch]

        # Convert images+questions into model inputs (pixel_values + tokenized question).
        inputs = processor(images=images, text=questions, return_tensors="pt", padding=True).to(device)

        # Tokenize answers into label IDs (short max_length to keep supervision compact).
        lab = processor.tokenizer(
            answers,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=16,
        ).input_ids.to(device)

        # Replace PAD tokens with -100 so cross-entropy ignores them (standard HF convention).
        lab = lab.masked_fill(lab == pad_id, -100)

        # Add labels into the same dict so the training step can pass them directly.
        inputs["labels"] = lab

        # Keep text answers for debugging (not used by the model forward pass).
        inputs["answers_text"] = answers
        return inputs

    return collate


@torch.no_grad()
def evaluate(model, processor, split, device: str, limit_n: int = 0):
    """
    Quick evaluation loop:
    generates answers with the current model, normalizes predictions, constrains closed-ended outputs,
    and returns metrics computed by eval_from_strings().
    """
    model.eval()
    n = len(split) if not limit_n else min(len(split), limit_n)

    preds, gts, qtypes = [], [], []
    for i in tqdm(range(n), desc="eval", leave=False):
        ex = split[i]

        # Read example fields.
        img = ex["image"]
        q = ex["question"]
        gt = norm_ans(ex["answer"])
        qtype = infer_qtype_from_gt(gt)

        # Generate an answer from BLIP.
        inputs = processor(img, q, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=10)

        # Decode and normalize prediction text.
        pred = processor.decode(out[0], skip_special_tokens=True)
        pred = norm_ans(pred)

        # For yes/no, force prediction into a constrained set (helps stability).
        if qtype == "closed":
            pred = constrain_closed_prediction(pred)

        preds.append(pred)
        gts.append(gt)
        qtypes.append(qtype)

    return eval_from_strings(gts, preds, qtypes)


def main():
    # ---- CLI arguments ----
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="Salesforce/blip-vqa-base")  # base BLIP checkpoint
    ap.add_argument("--out_dir", type=str, default="outputs/blip_lora")           # output directory for adapters
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)

    # LoRA hyperparameters.
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.05)

    # Optional caps for quick runs.
    ap.add_argument("--train_n", type=int, default=0)  # if >0, only train on first N examples
    ap.add_argument("--eval_n", type=int, default=0)   # if >0, only eval on first N examples
    args = ap.parse_args()

    # ---- Reproducibility + device ----
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare output directory.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load dataset ----
    ds = load_dataset("flaviagiammarino/vqa-rad")
    train_split = ds["train"]
    test_split = ds["test"]

    # Optionally restrict training size for debugging / fast experiments.
    if args.train_n and args.train_n > 0:
        train_split = train_split.select(range(min(args.train_n, len(train_split))))

    # ---- Load BLIP processor + base model ----
    processor = BlipProcessor.from_pretrained(args.model_name)
    base = BlipForQuestionAnswering.from_pretrained(args.model_name)

    # ---- Configure LoRA (PEFT) ----
    # Target_modules ["q","v"] typically map to attention projection layers in many transformer blocks.
    lora_cfg = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",                 # only train LoRA weights (no bias params)
        target_modules=["q", "v"],    # apply LoRA to attention projections (common approach)
        task_type="SEQ_2_SEQ_LM",     # PEFT task label (keeps adapter behavior consistent)
    )

    # Wrap base model with LoRA adapters (only adapter weights are trainable).
    model = get_peft_model(base, lora_cfg).to(device)

    # ---- Build DataLoader ----
    train_ds = VQARadBlipDataset(train_split)
    collate = make_collate(processor, device)
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
    )

    # AdamW is a standard optimizer choice for transformer finetuning.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Store per-epoch logs (loss + eval metrics) for later reporting.
    history = []
    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0

        # ---- Training loop ----
        for batch in tqdm(loader, desc=f"train epoch {ep}"):
            opt.zero_grad(set_to_none=True)

            # BLIPForQuestionAnswering expects these named arguments:
            # pixel_values (image tensor), input_ids/attention_mask (question tokens), labels (answer tokens).
            out = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            loss = out.loss
            loss.backward()
            opt.step()

            # Track running average loss (weighted by batch size).
            bs = batch["input_ids"].size(0)
            total_loss += float(loss.item()) * bs
            n_seen += bs

        avg_loss = total_loss / max(n_seen, 1)

        # ---- Evaluate after each epoch ----
        metrics = evaluate(model, processor, test_split, device, limit_n=args.eval_n)

        # Log everything in a single dict for easy JSON export.
        row = {"epoch": ep, "train_loss": avg_loss, **metrics}
        history.append(row)
        print(json.dumps(row, indent=2))

    # ---- Save adapters + processor + logs ----
    # For PEFT models, save_pretrained writes the adapter weights/config (not the full base weights).
    model.save_pretrained(str(out_dir))
    processor.save_pretrained(str(out_dir))
    (out_dir / "train_log.json").write_text(json.dumps(history, indent=2))

    print(f"\nSaved LoRA adapters + processor to: {out_dir}")


if __name__ == "__main__":
    main()
