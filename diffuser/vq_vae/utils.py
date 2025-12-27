import os
import json
import random
from typing import List

import numpy as np
import torch
from preprocessing.fuzzy_embedding import EMBEDDINGS_JSONL

NPY_PATH = "embeddings.npy"

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def l2_normalize(x: np.ndarray, eps: float = 1e-12):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)

def ensure_embeddings_npy(npy_path: str, l2_norm: bool = True) -> np.ndarray:
    if os.path.exists(npy_path):
        arr = np.load(npy_path)
        return arr.astype(np.float32, copy=False)

    # Collect all 1D vectors from the JSONL, preserving order.
    vectors: List[np.ndarray] = []
    dim: int = -1
    rows_total = 0
    empty_rows = 0
    empty_turns = 0
    turns_total = 0

    for obj in read_jsonl(EMBEDDINGS_JSONL):
        rows_total += 1
        row_vectors_before = len(vectors)

        # New format: list of turns, each with a list of sentence embeddings
        if "conversation_turns" in obj:
            for turn in obj.get("conversation_turns", []):
                turns_total += 1
                turn_vectors_before = len(vectors)
                for emb in turn.get("embeddings", []):
                    vec = np.asarray(emb, dtype=np.float32)
                    if vec.ndim != 1:
                        raise ValueError(f"Expected 1D embedding vector, got shape {vec.shape}")
                    if dim == -1:
                        dim = vec.shape[0]
                    elif vec.shape[0] != dim:
                        raise ValueError(f"Inconsistent embedding dim: expected {dim}, got {vec.shape[0]}")
                    vectors.append(vec)
                if len(vectors) == turn_vectors_before:
                    empty_turns += 1
        # Legacy format: single embedding per row under 'embeddings'
        elif "embeddings" in obj:
            vec = np.asarray(obj["embeddings"], dtype=np.float32)
            if vec.ndim != 1:
                raise ValueError(f"Expected 1D embedding vector, got shape {vec.shape}")
            if dim == -1:
                dim = vec.shape[0]
            elif vec.shape[0] != dim:
                raise ValueError(f"Inconsistent embedding dim: expected {dim}, got {vec.shape[0]}")
            vectors.append(vec)
            turns_total += 1
        if len(vectors) == row_vectors_before:
            empty_rows += 1

    if not vectors:
        raise ValueError("No embeddings found in jsonl")

    arr = np.stack(vectors, axis=0).astype(np.float32, copy=False)  # [N, D]

    print(
        f"[embeddings] rows={rows_total} turns={turns_total} vectors={len(vectors)} "
        f"dim={dim if dim != -1 else 'unknown'} empty_rows={empty_rows} empty_turns={empty_turns}"
    )

    if empty_rows or empty_turns:
        print("[embeddings] warning: some rows/turns had no embeddings; they were skipped")

    if l2_norm:
        arr = l2_normalize(arr)

    dir_name = os.path.dirname(npy_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    np.save(npy_path, arr)
    return arr

def read_jsonl(jsonl_path: str) -> List[dict]:
    rows: List[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def write_jsonl(jsonl_path: str, rows: List[dict]) -> None:
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# basically measure how many codes in the codebook are effectively used
@torch.no_grad()
def codebook_perplexity(indices: torch.Tensor, num_codes: int) -> float:
    if indices.numel() == 0:
        return 0.0
    counts = torch.bincount(indices, minlength=num_codes).float()
    probs = counts / counts.sum().clamp_min(1.0)
    entropy = -(probs * (probs.clamp_min(1e-12).log())).sum()
    return float(torch.exp(entropy).item())
