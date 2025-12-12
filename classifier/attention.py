from typing import Dict

import torch
from torch.nn.functional import softmax
from transformers import PreTrainedTokenizerBase, PreTrainedModel, AutoTokenizer, AutoModelForSequenceClassification

def _normalize_importance(importance: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = importance * mask
    total = masked.sum() + 1e-12
    return masked / total

def attention_importance(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, text: str):
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    model.eval()

    with torch.no_grad():
        outputs = model(**encoded, output_attentions=True)

    attentions = outputs.attentions
    last_layer = attentions[-1]  # shape: (batch, heads, seq, seq)
    
    # Mean over heads, take CLS row (index 0) as a simple attribution signal.
    cls_attn = last_layer.mean(dim=1)[0, 0]  # (seq,)

    attention_mask = encoded["attention_mask"][0].float()  # type: ignore[index]
    scores = _normalize_importance(cls_attn, attention_mask)
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])  # type: ignore[index]
    return tokens, scores.cpu().tolist()

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
    attention_mask = encoded["attention_mask"][0].float()  # type: ignore[index]
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
