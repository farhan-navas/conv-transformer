## conv-transformer

Bare-bones scaffold to fine-tune `roberta-base` for 3-class dialogue outcome classification with attention-based attribution.

### Setup

1. Install deps (uv/pip):
   - `uv sync`

### Train

```
uv run main.py
```

Printed metrics include validation macro-F1 (best epoch) and all test metrics inside metrics.jsonl.

### Attention

Use `conv_transformer.attention.attention_importance(model, tokenizer, text)` to get tokens and normalized attention scores from the CLS token.
