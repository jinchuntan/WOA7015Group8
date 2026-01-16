# src/finetune_blip_lora.py
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datasets import load_dataset
from transformers import BlipForQuestionAnswering, BlipProcessor

from .metrics import eval_from_strings
from .text_utils import norm_ans, infer_qtype_from_gt
from .lora_utils import (
    find_linear_module_names,
    inject_lora,
    mark_only_lora_trainable,
    extract_lora_state_dict,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


class VQARadRaw(Dataset):
    def __init__(self, split):
        self.split = split

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        ex = self.split[idx]
        img = ex["image"].convert("RGB")
        q = str(ex["question"])
        a = str(ex["answer"])
        return img, q, a


class BlipVQACollator:
    """
    Builds BLIP inputs and safe decoder inputs/labels.

    Critical detail (fixes your IndexError):
    - We provide decoder_input_ids explicitly.
    - labels can contain -100 for loss masking, but decoder_input_ids never contains -100.
    """
    def __init__(self, processor: BlipProcessor, max_answer_len: int = 12):
        self.processor = processor
        self.tok = processor.tokenizer
        self.max_answer_len = int(max_answer_len)

        # robust special token ids
        self.pad = self.tok.pad_token_id
        if self.pad is None:
            # fallback
            self.pad = 0

        self.bos = self.tok.cls_token_id
        if self.bos is None:
            self.bos = self.tok.bos_token_id
        if self.bos is None:
            self.bos = self.tok.sep_token_id
        if self.bos is None:
            self.bos = self.pad

        self.eos = self.tok.sep_token_id
        if self.eos is None:
            self.eos = self.tok.eos_token_id
        if self.eos is None:
            self.eos = self.pad

    def _build_answer_tensors(self, answers: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          decoder_input_ids: [B, T]
          decoder_attention_mask: [B, T]
          labels: [B, T]  (pad positions are -100)
        """
        # Tokenize WITHOUT special tokens; we add EOS ourselves
        tok_out = self.tok(
            answers,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=self.max_answer_len - 1,  # reserve 1 for EOS
        )
        ids_list = tok_out["input_ids"]

        B = len(ids_list)
        T = self.max_answer_len

        label_ids = torch.full((B, T), self.pad, dtype=torch.long)
        for i, ids in enumerate(ids_list):
            ids = ids[: T - 1]
            ids = ids + [self.eos]
            label_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

        decoder_input_ids = torch.full((B, T), self.pad, dtype=torch.long)
        decoder_input_ids[:, 0] = self.bos
        decoder_input_ids[:, 1:] = label_ids[:, :-1]

        decoder_attention_mask = (decoder_input_ids != self.pad).long()

        labels = label_ids.clone()
        labels[labels == self.pad] = -100

        return decoder_input_ids, decoder_attention_mask, labels

    def __call__(self, batch):
        imgs, qs, ans = zip(*batch)

        enc = self.processor(
            images=list(imgs),
            text=list(qs),
            return_tensors="pt",
            padding=True,
        )

        decoder_input_ids, decoder_attention_mask, labels = self._build_answer_tensors(list(ans))

        enc["decoder_input_ids"] = decoder_input_ids
        enc["decoder_attention_mask"] = decoder_attention_mask
        enc["labels"] = labels
        return enc


@torch.no_grad()
def eval_blip(model, processor, split, device: str, max_new_tokens: int = 10, num_beams: int = 3, limit_n: int = 0):
    model.eval()
    preds, gts, qtypes = [], [], []

    n = len(split)
    if limit_n and limit_n > 0:
        n = min(n, limit_n)

    for i in tqdm(range(n), desc="val eval"):
        ex = split[i]
        img = ex["image"].convert("RGB")
        q = str(ex["question"])
        gt = norm_ans(str(ex["answer"]))
        qt = infer_qtype_from_gt(gt)

        if qt == "closed":
            inputs = processor(img, q, return_tensors="pt").to(device)
            yes_ids = processor.tokenizer("yes", return_tensors="pt").input_ids.to(device)
            no_ids  = processor.tokenizer("no",  return_tensors="pt").input_ids.to(device)
            loss_yes = model(**inputs, labels=yes_ids).loss.item()
            loss_no  = model(**inputs, labels=no_ids).loss.item()
            pred = "yes" if loss_yes < loss_no else "no"
        else:
            inputs = processor(img, q, return_tensors="pt").to(device)
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                early_stopping=True,
            )
            pred = processor.decode(out[0], skip_special_tokens=True)
            pred = norm_ans(pred)

        preds.append(pred)
        gts.append(gt)
        qtypes.append(qt)

    return eval_from_strings(gts, preds, qtypes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="Salesforce/blip-vqa-base")
    ap.add_argument("--out_dir", type=str, default="outputs/blip_lora")

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max_answer_len", type=int, default=12)

    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # Where to inject LoRA
    ap.add_argument("--lora_include", type=str, default="text_decoder")
    ap.add_argument("--lora_suffixes", type=str, default="query,key,value")

    # eval knobs
    ap.add_argument("--val_limit", type=int, default=0)  # 0 = full val
    ap.add_argument("--val_max_new_tokens", type=int, default=10)
    ap.add_argument("--val_num_beams", type=int, default=3)

    # debug
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_val_samples", type=int, default=0)

    args = ap.parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("flaviagiammarino/vqa-rad")
    train_split = ds["train"]
    if "validation" in ds:
        val_split = ds["validation"]
    elif "val" in ds:
        val_split = ds["val"]
    else:
        split = train_split.train_test_split(test_size=0.1, seed=args.seed)
        train_split = split["train"]
        val_split = split["test"]

    if args.max_train_samples and args.max_train_samples > 0:
        train_split = train_split.select(range(min(args.max_train_samples, len(train_split))))
    if args.max_val_samples and args.max_val_samples > 0:
        val_split = val_split.select(range(min(args.max_val_samples, len(val_split))))

    processor = BlipProcessor.from_pretrained(args.model_name)
    model = BlipForQuestionAnswering.from_pretrained(args.model_name)

    # -------- LoRA injection (manual, stable) --------
    suffixes = [s.strip() for s in args.lora_suffixes.split(",") if s.strip()]
    target_names = find_linear_module_names(
        model,
        include_substring=args.lora_include,
        suffixes=suffixes,
    )

    if len(target_names) == 0:
        raise RuntimeError(
            f"No target Linear modules found for LoRA. include='{args.lora_include}', suffixes={suffixes}. "
            f"Try include='text_decoder' and suffixes='query,key,value'."
        )

    replaced = inject_lora(
        model,
        module_names=target_names,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

    trainable = mark_only_lora_trainable(model)
    print(f"[LoRA] Replaced {replaced} Linear layers with LoRA.")
    print(f"[LoRA] Trainable params: {trainable:,}")
    print(f"[LoRA] Example targets (first 10):")
    for n in target_names[:10]:
        print("  -", n)

    model.to(device)

    train_ds = VQARadRaw(train_split)
    val_ds = VQARadRaw(val_split)

    collator = BlipVQACollator(processor, max_answer_len=args.max_answer_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collator)

    # Optimizer only over trainable params (LoRA)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)

    history = []
    best_val = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        steps = 0

        opt.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"train {epoch}/{args.epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            # forward (safe: decoder_input_ids provided explicitly)
            try:
                out = model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    decoder_input_ids=batch["decoder_input_ids"],
                    decoder_attention_mask=batch["decoder_attention_mask"],
                    labels=batch["labels"],
                )
            except TypeError:
                # if your transformers build doesn't accept decoder_attention_mask
                out = model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    decoder_input_ids=batch["decoder_input_ids"],
                    labels=batch["labels"],
                )

            loss = out.loss / max(1, args.grad_accum)
            loss.backward()

            running += float(loss.item())
            steps += 1

            if steps % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            pbar.set_postfix(loss=float(loss.item()))

        # epoch eval
        val_metrics = eval_blip(
            model=model,
            processor=processor,
            split=val_split,
            device=device,
            max_new_tokens=args.val_max_new_tokens,
            num_beams=args.val_num_beams,
            limit_n=args.val_limit,
        )

        rec = {
            "epoch": epoch,
            "train_loss": float(running / max(1, steps)),
            "val_overall_exact_match": float(val_metrics["overall_exact_match"]),
            "val_closed_accuracy": float(val_metrics["closed_accuracy"]),
            "val_open_exact_match": float(val_metrics["open_exact_match"]),
            "val_open_token_f1": float(val_metrics["open_token_f1"]),
            "val_open_rougeL_f1": float(val_metrics["open_rougeL_f1"]),
        }
        history.append(rec)
        (out_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(rec, indent=2))

        # save best by val overall EM
        if rec["val_overall_exact_match"] > best_val:
            best_val = rec["val_overall_exact_match"]

            lora_sd = extract_lora_state_dict(model)
            torch.save(lora_sd, out_dir / "lora.pt")

            cfg = {
                "base_model_name": args.model_name,
                "lora_include": args.lora_include,
                "lora_suffixes": suffixes,
                "target_module_names": target_names,
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "max_answer_len": args.max_answer_len,
            }
            (out_dir / "lora_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

            # also save processor for safety/repro
            processor.save_pretrained(str(out_dir))
            print(f"[saved] Best LoRA checkpoint to: {out_dir}")

    print(f"\nDone. Best val overall EM = {best_val:.4f}")
    print(f"LoRA saved at: {out_dir/'lora.pt'} and {out_dir/'lora_config.json'}")


if __name__ == "__main__":
    main()
