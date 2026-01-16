# src/text_utils.py
# Written by: Nigel Tan Jin Chun, 24054001
# Last Modified: 10/1/2026
# Purpose: Provide shared text utilities for the VQA pipeline, including answer normalization for
#          exact-match evaluation, simple tokenization, question vocabulary building/encoding, and
#          basic heuristics to infer question type (closed vs open).

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List, Tuple


# Translation table that converts punctuation characters into spaces (used for normalization/tokenization).
_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})


def norm_ans(s: str) -> str:
    """Normalize answers for string-based evaluation (lowercase, remove punctuation, collapse whitespace)."""
    if s is None:
        return ""
    s = str(s).strip().lower()        # standardize type + case + surrounding whitespace
    s = s.translate(_PUNCT_TABLE)     # replace punctuation with spaces
    s = re.sub(r"\s+", " ", s).strip()  # collapse multiple spaces into one
    return s


def simple_tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer (lowercase + punctuation-to-space + split)."""
    if text is None:
        return []
    text = str(text).lower().strip()     # normalize type + case + whitespace
    text = text.translate(_PUNCT_TABLE)  # treat punctuation as separators
    toks = re.sub(r"\s+", " ", text).strip().split(" ")  # normalize spacing then split
    toks = [t for t in toks if t]        # drop empty tokens
    return toks


def build_question_vocab(
    questions: List[str],
    min_freq: int = 1,
    max_size: int = 20000,
) -> Dict[str, int]:
    """
    Build a question vocabulary mapping token -> id.

    Reserved IDs:
      <pad> = 0  (padding)
      <unk> = 1  (unknown token)
    """
    # Count tokens across all questions.
    cnt = Counter()
    for q in questions:
        cnt.update(simple_tokenize(q))

    # Initialize vocab with reserved tokens.
    vocab = {"<pad>": 0, "<unk>": 1}

    # Add tokens in descending frequency order until min_freq/max_size limits are reached.
    for tok, f in cnt.most_common():
        if f < min_freq:
            break
        if tok in vocab:
            continue
        vocab[tok] = len(vocab)
        if len(vocab) >= max_size:
            break

    return vocab


def encode_question(q: str, vocab: Dict[str, int], max_len: int) -> Tuple[List[int], int]:
    """
    Convert a question string into:
      - a fixed-length list of token IDs (padded/truncated to max_len)
      - the true (unpadded) length capped at max_len
    """
    toks = simple_tokenize(q)
    ids = [vocab.get(t, vocab["<unk>"]) for t in toks]  # map tokens to IDs with <unk> fallback

    # True sequence length after truncation.
    q_len = min(len(ids), max_len)

    # Truncate and then pad to exactly max_len.
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids = ids + [vocab["<pad>"]] * (max_len - len(ids))

    return ids, q_len


def infer_qtype_from_gt(gt_answer_norm: str) -> str:
    """Ground-truth based type: closed if normalized answer is yes/no, else open."""
    a = norm_ans(gt_answer_norm)
    return "closed" if a in {"yes", "no"} else "open"


def infer_qtype_from_question(question: str) -> str:
    """
    Heuristic type inference (no GT): treat questions that start like yes/no prompts as closed.
    Note: this is optional; many evaluations still report closed/open based on GT labels.
    """
    q = (question or "").strip().lower()

    # Common auxiliary/modal starters for yes/no questions.
    starters = ("is", "are", "was", "were", "do", "does", "did", "has", "have", "can", "could", "will", "would")
    if q.startswith(starters):
        return "closed"
    return "open"
