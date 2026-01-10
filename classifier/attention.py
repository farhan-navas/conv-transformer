from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .data import DataConfig


@dataclass
class AttentionConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model_dir: str = "model_out"
    output_path: str = "attention_scores.txt"
    sample_size: int = 20
    max_length: int = 512
    layer: int = -1
    seed: int = 42


def _normalize_importance(importance: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = importance * mask
    total = masked.sum() + 1e-12
    return masked / total

def _mask_special(encoded: Dict[str, torch.Tensor], tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    attention_mask = encoded["attention_mask"][0].clone().float()
    input_ids = encoded["input_ids"][0]

    special: List[int] = []
    for t in (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id):
        if isinstance(t, int):
            special.append(t)
        elif isinstance(t, Iterable):
            special.extend([tt for tt in t if isinstance(tt, int)])
    if not special:
        return attention_mask

    special_t = torch.tensor(special, device=input_ids.device)
    is_special = (input_ids.unsqueeze(-1) == special_t).any(dim=-1)
    attention_mask[is_special] = 0.0
    return attention_mask


def _sample_indices(n: int, sample_size: int, seed: int) -> List[int]:
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng)
    return perm[:min(sample_size, n)].tolist()

def predict_with_attention(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    id2label: Dict[int, str] | None = None,
    max_length: int = 512,
    layer: int = -1,
) -> Dict:
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    model.eval()
    with torch.no_grad():
        outputs = model(**encoded, output_attentions=True)
    logits = outputs.logits
    probs = softmax(logits, dim=-1)[0].cpu().tolist()
    pred_id = int(torch.argmax(logits, dim=-1).item())
    pred_label = id2label.get(pred_id, str(pred_id)) if id2label else str(pred_id)

    attentions = outputs.attentions
    last_layer = attentions[layer]
    cls_attn = last_layer.mean(dim=1)[0, 0]
    attention_mask = _mask_special(encoded, tokenizer)  # type: ignore
    scores = _normalize_importance(cls_attn, attention_mask)
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])  # type: ignore[index]

    return {
        "pred_id": pred_id,
        "pred_label": pred_label,
        "probs": probs,
        "tokens": tokens,
        "attention_scores": scores.cpu().tolist(),
    }


def run_attention_dump(config: AttentionConfig) -> None:
    df = pd.read_csv(config.data.csv_path)
    id2label = {v: k for k, v in config.data.label_map.items()}

    tokenizer = AutoTokenizer.from_pretrained(config.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_dir)

    out_path = Path(config.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, mode="w", encoding="utf-8") as f:
        for i in _sample_indices(len(df), sample_size=config.sample_size, seed=config.seed):
            ex = df.iloc[i]
            text = ex[config.data.text_column]
            label = ex[config.data.label_column]
            res = predict_with_attention(
                model,
                tokenizer,
                text,
                id2label=id2label,
                max_length=config.max_length,
                layer=config.layer,
            )

            token_scores = list(zip(res["tokens"], res["attention_scores"]))

            f.write(
                f"Example {i}: gold={str(label)} pred={res['pred_label']} probs={res['probs']}\n"
                f"Tokens+attn: {token_scores}\n"
                f"{'=' * 60}\n\n"
            )

