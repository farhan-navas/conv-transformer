## conv-transformer

Unified entrypoint for classifier fine-tuning, attention dumps, speaker labelling, fuzzy embedding creation, VQ-VAE training, and codebook export.

### Config

- Edit `config.yaml`.
- `core.run_version` sets the output folder under `runs/` (e.g., `v0.1`, `v0.2`).
- `core.task` chooses which job to run: `classifier_train`, `attention`, `create_embeddings`, `label_speakers`, `vqvae_train`, `export_codes`.
- ALL settings + data live in their own sections.

### Run

- Classifier train (uses `core.task` by default):
  ```
  uv run main.py --config config.yaml --task classifier_train --run-version v0.1
  ```
- Attention dump (reads saved classifier model):
  ```
  uv run main.py --config config.yaml --task attention --run-version v0.1
  ```
- VQ-VAE train:
  ```
  uv run main.py --config config.yaml --task vqvae_train --run-version v0.1
  ```
- Export code assignments:
  ```
  uv run main.py --config config.yaml --task export_codes --run-version v0.1
  ```

### Outputs

- Everything is written under `runs/<run_version>/`:
  - `classifier/` holds the fine-tuned model and `metrics.jsonl`.
  - `attention/` holds attention dumps.
  - `vqvae/` holds checkpoints, embeddings, and exported cluster IDs.

### Module map

- `classifier/data.py`: dataset config, splits, label map.
- `classifier/train.py`: classifier training.
- `classifier/attention.py`: sample attention dump from a saved classifier.
- `diffuser/preprocessing/label_speakers.py`: label speakers in data with confidence score
- `diffuser/preprocessing/fuzzy_embedding.py`: create sentence embeddings from dataset
- `diffuser/vq_vae/train.py`: VQ-VAE training on dataset
- `diffuser/vq_vae/export_codes.py`: export code assignments per conversation turn.

## Conversation Dataset Pre-processing

This section is how we initially plan to clean the conversation dataset. The dataset initially starts off an input CSV file with the following required columns that need to be processed: ...

Preprocessing modules explanation under `diffuser/preprocessing`:

- `data_prep.py`: cleans/splits conversation and channel texts, labels utterances to speakers via similarity, merges consecutive turns, builds per-turn embeddings. Very simple manual preprocessing.
- `fuzzy_embedding.py`: loads CSV rows, labels speakers in conversation vs channel references, merges consecutive turns, builds per-turn sentence embeddings, and writes conversation_turns/embeddings JSONL plus outcome stats.
- `label_speakers.py`: cleans/normalizes channel transcripts, predicts Agent/Donor roles with a pre-trained classifier, and writes a human-readable CSV with role labels and confidences.

## Attention

Bare-bones scaffold to fine-tune `roberta-base` for 3-class dialogue outcome classification with attention-based attribution (inside `/classifier`). This was done to check if we could find any correlation between certain tokens and the target disposition, and as expected, since dataset is incomplete as of now, model consistently uses "nice"/callback/rejection words (based on the specific scenario) and punctuation as "attention sinks", i think.

Printed metrics can be found in `attention_scores_sorted.jsonl`.
Conversation dataset metrics can be found inside `overall_metrics.json`, ...: TODO: describe data shape as well
