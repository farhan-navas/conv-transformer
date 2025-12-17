from typing import Dict

import torch
from torch.nn.functional import softmax
from transformers import PreTrainedTokenizerBase, PreTrainedModel, AutoTokenizer, AutoModelForSequenceClassification

def _normalize_importance(importance: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = importance * mask
    total = masked.sum() + 1e-12
    return masked / total

def _mask_special(encoded: Dict[str, torch.Tensor], tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    attention_mask = encoded["attention_mask"][0].clone().float()
    input_ids = encoded["input_ids"][0]

    special = [ # specifically for roberta
        t for t in (
            tokenizer.bos_token_id,  # <s>
            tokenizer.eos_token_id,  # </s>
            tokenizer.pad_token_id,  # <pad>
        )
        if t is not None
    ]
    if not special:
        return attention_mask

    special_t = torch.tensor(special, device=input_ids.device)
    is_special = (input_ids.unsqueeze(-1) == special_t).any(dim=-1)
    attention_mask[is_special] = 0.0
    return attention_mask

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

out = predict_with_attention(mdl, tok, "Thank you so much for the donation bhaiya!", id2label=id2label)
print(out["pred_label"], out["probs"])
print(list(zip(out["tokens"], out["attention_scores"])))
