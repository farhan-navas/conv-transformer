from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CodeModelConfig:
    vocab_size: int
    pad_id: int
    cls_id: Optional[int]
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    max_length: int = 512
    num_labels: int = 3


class CodeClassifier(nn.Module):
    def __init__(self, cfg: CodeModelConfig):
        super().__init__()
        self.cfg = cfg

        effective_vocab = max(
            cfg.vocab_size,
            (cfg.pad_id + 1) if cfg.pad_id is not None else 0,
            (cfg.cls_id + 1) if cfg.cls_id is not None else 0,
        )

        self.token_embed = nn.Embedding(effective_vocab, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_embed = nn.Embedding(cfg.max_length, cfg.d_model)
        self.layers = nn.ModuleList(
            [
                AttentionEncoderLayer(
                    d_model=cfg.d_model,
                    nhead=cfg.n_heads,
                    dim_feedforward=cfg.d_model * 4,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attentions: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        # input_ids: [B, T], attention_mask: [B, T] where 1=keep, 0=pad
        B, T = input_ids.shape
        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)

        x = self.token_embed(input_ids) + self.pos_embed(pos_ids)
        key_padding = attention_mask == 0
        attentions: list[torch.Tensor] = []
        enc = x
        for layer in self.layers:
            enc, attn = layer(enc, src_key_padding_mask=key_padding, need_weights=return_attentions)
            if return_attentions and attn is not None:
                attentions.append(attn)

        # CLS-style pooling: take first token
        pooled = enc[:, 0, :]
        pooled = self.norm(pooled)
        logits = self.head(pooled)
        if return_attentions:
            return logits, attentions
        return logits


class AttentionEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Self-attention block with optional weight return
        attn_output, attn_weights = self.self_attn(
            src,
            src,
            src,
            key_padding_mask=src_key_padding_mask,
            need_weights=need_weights,
        )
        src2 = self.dropout1(attn_output)
        src = self.norm1(src + src2)
        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src2 = self.dropout2(ff)
        src = self.norm2(src + src2)
        return src, attn_weights
