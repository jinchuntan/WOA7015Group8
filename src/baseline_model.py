# src/baseline_model.py
# Written by: Nigel Tan Jin Chun, 24054001
# Last Modified: 10/1/2026
# Purpose: Define a baseline VQA model that encodes images with a (optionally frozen) ResNet-50,
#          encodes questions with an embedding + bidirectional LSTM, concatenates both features,
#          and predicts an answer class from a fixed answer vocabulary.

from __future__ import annotations  # allow forward references in type hints

import torch
import torch.nn as nn
from torchvision import models


class BaselineVQAModel(nn.Module):
    """
    Baseline architecture:
    - Image encoder: ResNet50 backbone (fc removed), frozen by default
    - Question encoder: token embedding + BiLSTM
    - Fusion: concatenate image + question features, then MLP classifier to answer vocab

    IMPORTANT: Module name "fusion" is kept stable so checkpoints stay compatible.
    """

    def __init__(
        self,
        n_answers: int,
        vocab_size: int,
        emb_dim: int = 200,
        lstm_hidden: int = 256,
        freeze_cnn: bool = True,
    ):
        super().__init__()

        # ---- Image encoder (ResNet50) ----
        # Load an ImageNet-pretrained ResNet50 and remove the final classification layer.
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        resnet.fc = nn.Identity()  # output becomes the 2048-d penultimate feature vector
        self.cnn = resnet

        # Optionally freeze the CNN so only the text/fusion layers train (common for small datasets).
        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False

        # ---- Question encoder (Embedding + BiLSTM) ----
        # padding_idx=0 ensures the padding token embedding is not updated and stays neutral.
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        # Bidirectional LSTM produces two final hidden states (forward + backward).
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=lstm_hidden,
            batch_first=True,    # inputs are [B, L, E]
            bidirectional=True,  # output hidden states are doubled (2 directions)
        )

        # ResNet feature dim (2048) + BiLSTM feature dim (2 * lstm_hidden)
        fused_dim = 2048 + (2 * lstm_hidden)

        # ---- Fusion + classification head ----
        # Keep this module name stable ("fusion") to remain checkpoint-compatible.
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_answers),  # output logits over answer vocabulary
        )

    def forward(self, images: torch.Tensor, q_ids: torch.Tensor, q_lens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        images: [B, 3, H, W]   input images
        q_ids:  [B, L]         token IDs for questions (padded)
        q_lens: [B]            true (unpadded) lengths for each question
        """
        # Encode images into a single feature vector per image: [B, 2048]
        img_feat = self.cnn(images)

        # Look up token embeddings for questions: [B, L, E]
        q_emb = self.emb(q_ids)

        # Pack padded sequences so the LSTM ignores padded tokens and uses correct lengths.
        # pack_padded_sequence expects lengths on CPU and in integer form.
        q_lens_cpu = q_lens.detach().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            q_emb,
            q_lens_cpu,
            batch_first=True,
            enforce_sorted=False,  # allow unsorted batches (common in DataLoader)
        )

        # Run BiLSTM; we only need the final hidden states from both directions.
        # h has shape [2, B, H] (2 directions, batch, hidden).
        _, (h, _) = self.lstm(packed)

        # Concatenate forward and backward final hidden states: [B, 2H]
        q_feat = torch.cat([h[0], h[1]], dim=1)

        # Fuse image + question features and classify: logits [B, n_answers]
        fused = torch.cat([img_feat, q_feat], dim=1)
        logits = self.fusion(fused)
        return logits
