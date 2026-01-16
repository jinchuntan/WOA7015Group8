# src/data_vqarad.py
# Written by: Nigel Tan Jin Chun, 24054001
# Last Modified: 10/1/2026
# Purpose: Load the VQA-RAD dataset splits and provide a PyTorch Dataset + collate function for the
#          baseline (CNN+BiLSTM) model, including image preprocessing, question tokenization/encoding,
#          optional filtering of out-of-vocabulary answers and optional tracking of original indices.

from __future__ import annotations  # allow forward references in type hints

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from PIL import Image
from torchvision import transforms

from .text_utils import encode_question, norm_ans  # question -> ids/length + answer normalization


# ImageNet normalization stats (match pretrained ResNet/vision backbones expectations)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def load_vqarad_splits(seed: int = 42, val_ratio: float = 0.1):
    """
    Load VQA-RAD from HuggingFace and always return (train, val, test).

    Some dataset versions may not include a dedicated validation split, so we create one by
    splitting the training set using val_ratio.
    """
    ds = load_dataset("flaviagiammarino/vqa-rad")  # HuggingFace dataset dict
    train = ds["train"]
    test = ds["test"]

    # Prefer an existing validation split if it exists under common keys.
    if "validation" in ds:
        val = ds["validation"]
    elif "val" in ds:
        val = ds["val"]
    else:
        # If no validation split exists, split the training set deterministically with a seed.
        tmp = train.train_test_split(test_size=val_ratio, seed=seed)
        train, val = tmp["train"], tmp["test"]

    return train, val, test


class VQARadBaselineDataset(Dataset):
    """
    PyTorch dataset wrapper for the baseline VQA pipeline.

    It:
      - applies standard image transforms (resize -> tensor -> ImageNet normalize)
      - normalizes answers with norm_ans()
      - optionally filters out examples whose answers are not in ans2id (OOV filtering)
      - encodes questions into fixed-length token id sequences + true length
      - optionally returns the original index within the split for traceability
    """

    def __init__(
        self,
        split,
        vocab_q: Dict[str, int],
        ans2id: Dict[str, int],
        max_q_len: int = 30,
        img_size: int = 224,
        filter_oov: bool = True,
        keep_orig_indices: bool = True,
    ):
        # Store references and config needed by __getitem__.
        self.split = split
        self.vocab_q = vocab_q
        self.ans2id = ans2id
        self.max_q_len = max_q_len
        self.filter_oov = filter_oov
        self.keep_orig_indices = keep_orig_indices

        # Image preprocessing to match typical CNN backbones (e.g., pretrained ResNet50).
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),              # fixed-size input
            transforms.ToTensor(),                                # [H,W,C] -> [C,H,W], scaled to [0,1]
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),   # normalize per channel
        ])

        # Build an index list so we can optionally drop examples with answers not in ans2id.
        # This keeps __len__ and __getitem__ consistent with the filtered set.
        self.indices: List[int] = []
        for i in range(len(split)):
            a = norm_ans(split[i]["answer"])  # normalize answer text for consistent mapping
            if (not filter_oov) or (a in ans2id):
                self.indices.append(i)

    def __len__(self):
        # Dataset length after optional OOV filtering.
        return len(self.indices)

    def __getitem__(self, idx: int):
        # Map the requested item index into the underlying split index.
        orig_i = self.indices[idx]
        ex = self.split[orig_i]

        # Load and standardize image format.
        img: Image.Image = ex["image"].convert("RGB")

        # Extract raw question and normalized answer.
        q = ex["question"]
        a = norm_ans(ex["answer"])

        # Encode question into fixed-length ids + true length (for packing/padding in the model).
        q_ids, q_len = encode_question(q, self.vocab_q, self.max_q_len)

        # Return tensors for model input + raw strings for logging/analysis.
        item = {
            "image": self.tf(img),                              # float tensor [3, img_size, img_size]
            "q_ids": torch.tensor(q_ids, dtype=torch.long),     # long tensor [max_q_len]
            "q_len": torch.tensor(q_len, dtype=torch.long),     # long tensor scalar
            "answer": a,                                        # normalized answer string
            "question": q,                                      # original question string
        }

        # Optionally include original split index for traceability (e.g., subset evaluation).
        if self.keep_orig_indices:
            item["orig_index"] = orig_i

        return item


def collate_baseline(batch: List[dict]) -> dict:
    """
    Collate function for DataLoader:
    stacks image/question tensors and keeps answer/question strings as lists.
    """
    images = torch.stack([b["image"] for b in batch], dim=0)  # [B, 3, H, W]
    q_ids = torch.stack([b["q_ids"] for b in batch], dim=0)   # [B, L]
    q_lens = torch.stack([b["q_len"] for b in batch], dim=0)  # [B]

    # Keep text fields as Python lists for evaluation/logging (not used directly by the model).
    answers = [b["answer"] for b in batch]
    questions = [b["question"] for b in batch]

    # Preserve original indices if available; otherwise use -1 placeholder.
    orig_idx = [b.get("orig_index", -1) for b in batch]

    return {
        "images": images,
        "q_ids": q_ids,
        "q_lens": q_lens,
        "answers": answers,
        "questions": questions,
        "orig_index": orig_idx,
    }
