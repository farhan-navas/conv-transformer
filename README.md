## conv-transformer

Bare-bones scaffold to fine-tune `roberta-base` for 3-class dialogue outcome classification with attention-based attribution (inside `/classifier`). This was done to check if we could find any correlation between certain tokens and the target disposition, and as expected, since dataset is incomplete as of now, model consistently uses "nice"/callback/rejection words (based on the specific scenario) and punctuation as "attention sinks", i think.

### Setup

1. Install deps (uv/pip):
   - `uv sync`

### Train

```
uv run main.py
```

Printed metrics can be found in `attention_scores_sorted.jsonl`.
Conversation dataset metrics can be found inside `overall_metrics.json`, describes the split of data that we have like

### Attention (Initial)

Just run `conv_transformer.attention` to get tokens and normalized attention scores from the CLS token. Classifier modules explanation under `classifier/`:

- `data.py`: loads CSVs, maps labels, build splits, and provide the tokenizer collated with optional class weights.
- `train.py`: training entrypoint, use weighted cross-entropy with AdamW optimizer, tracks macro-F1, and saves the best checkpoint/model artifacts to `model_out`
- `attention.py`: run saved model on samples and emit tokens with CLS attention scores (special tokens masked) for quick attention score inspection.

## Conversation Dataset Pre-processing

This section is how we initially plan to clean the conversation dataset. The dataset initially starts off an input CSV file with the following required columns that need to be processed: ...

Preprocessing modules explanation under `diffuser/preprocessing`:

- `data_prep.py`: cleans/splits conversation and channel texts, labels utterances to speakers via similarity, merges consecutive turns, builds per-turn embeddings. Very simple manual preprocessing.
- `fuzzy_embedding.py`: loads CSV rows, labels speakers in conversation vs channel references, merges consecutive turns, builds per-turn sentence embeddings, and writes conversation_turns/embeddings JSONL plus outcome stats.
- `label_speakers.py`: cleans/normalizes channel transcripts, predicts Agent/Donor roles with a pre-trained classifier, and writes a human-readable CSV with role labels and confidences.
