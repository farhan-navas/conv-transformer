## conv-transformer

Bare-bones scaffold to fine-tune `roberta-base` for 3-class dialogue outcome classification with attention-based attribution (inside `/classifier`). This was done to check if we could find any correlation between certain tokens and the target disposition, and as expected, since dataset is incomplete as of now, model consistently uses "nice"/callback/rejection words (based on the specific scenario) and punctuation as "attention sinks", i think.

### Setup

1. Install deps (uv/pip):
   - `uv sync`

### Train

```
uv run main.py
```

Printed metrics and cleaned data shape can be found inside `conversation_turns.json`, `attention_scores_sorted.jsonl`, `dataset_metrics.json` & `sentence_embeddings.json`.

### Attention

Just run `conv_transformer.attention` to get tokens and normalized attention scores from the CLS token.
