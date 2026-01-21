# Transformer Classifier

Codebook-based transformer classifier over conversation code IDs (cluster IDs) with optional attention dump for inspecting attention weights.

## Contents

- [classifier/train.py](classifier/train.py) — training loop for the code classifier
- [classifier/code_model.py](classifier/code_model.py) — model definition with attention-returning encoder layers
- [classifier/data.py](classifier/data.py) — data loaders for code sequences
- [classifier/attention.py](classifier/attention.py) — attention dump script

## Quick start

1. Train classifier (eg: via existing training entrypoint) to produce `model_out/model.pt` and ensure that `sentence_cluster_ids.jsonl` is available.
2. Generate attention scores:
   - After training, run:
     - `uv run main.py --task attention`
