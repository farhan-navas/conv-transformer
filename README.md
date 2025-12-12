## conv-transformer

Bare-bones scaffold to fine-tune `roberta-base` for 3-class dialogue outcome classification with attention-based attribution.

### Setup

1. Install deps (uv/pip):
   - `uv pip install -r pyproject.toml` (or `pip install -e .` if packaging)
2. Ensure your CSV has columns: `outcome` (negative|neutral|positive) and `transcript`.

### Train

```
uv run python main.py /path/to/data.csv --epochs 3 --batch-size 8
```

Printed metrics include validation macro-F1 (best epoch) and test metrics.

### Attribution (attention)

Use `conv_transformer.attention.attention_importance(model, tokenizer, text)` to get tokens and normalized attention scores from the CLS token.
