from __future__ import annotations

from typing import List, Tuple

import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel

def _normalize_importance(importance: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = importance * mask
    total = masked.sum() + 1e-12
    return masked / total


def attention_importance(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    max_length: int = 512,
    layer: int = -1,
) -> Tuple[List[str], List[float]]:
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    model.eval()

    with torch.no_grad():
        outputs = model(**encoded, output_attentions=True)

    attentions = outputs.attentions
    last_layer = attentions[layer]  # shape: (batch, heads, seq, seq)
    
    # Mean over heads, take CLS row (index 0) as a simple attribution signal.
    cls_attn = last_layer.mean(dim=1)[0, 0]  # (seq,)

    attention_mask = encoded["attention_mask"][0].float()  # type: ignore[index]
    scores = _normalize_importance(cls_attn, attention_mask)
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])  # type: ignore[index]
    return tokens, scores.cpu().tolist()
