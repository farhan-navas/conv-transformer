from typing import Dict, List, Iterable

import torch
import pandas as pd
from torch.nn.functional import softmax
from transformers import PreTrainedTokenizerBase, PreTrainedModel, AutoTokenizer, AutoModelForSequenceClassification
from data import DATA_COLUMN, LABEL_COLUMN

DATASET_SIZE = 30000
OUTPUT_FILE = "attention_scores.jsonl"

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

def _sample_indices(n: int, sample_size: int = 20) -> List[int]:
    rng = torch.Generator().manual_seed(42)
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
    attention_mask = _mask_special(encoded, tokenizer) # type: ignore
    scores = _normalize_importance(cls_attn, attention_mask)
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])  # type: ignore[index]

    return {
        "pred_id": pred_id,
        "pred_label": pred_label,
        "probs": probs,
        "tokens": tokens,
        "attention_scores": scores.cpu().tolist(),
    }

model_dir = "model_out"
id2label = {0: "Wrong Number", 1: "Callback", 2: "Promise to Pay"}
tok = AutoTokenizer.from_pretrained(model_dir)
mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)


def main():
    df = pd.read_csv("data.csv")
    with open(OUTPUT_FILE, mode="w", encoding="utf-8") as f:
        for i in _sample_indices(len(df), sample_size=20):
            ex = df.iloc[i]
            text = ex[DATA_COLUMN]
            label = ex[LABEL_COLUMN]
            res = predict_with_attention(mdl, tok, text, id2label=id2label)

            # Pair tokens with attention scores for readability
            token_scores = list(zip(res["tokens"], res["attention_scores"]))

            f.write(
                f"Example {i}: gold={str(label)} pred={res['pred_label']} probs={res['probs']}\n"
                f"Tokens+attn: {token_scores}\n"
                f"{'=' * 60}\n\n"
            )

if __name__ == "__main__":
    main()
