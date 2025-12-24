import os
import json
import random
from typing import List

import numpy as np
import torch
from preprocessing.fuzzy_embedding import EMBEDDINGS_JSONL

NPY_PATH = "sentence_embeddings.npy"

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

    embs: List[np.ndarray] = []
    for obj in read_jsonl(EMBEDDINGS_JSONL):
        vec = np.asarray(obj["embeddings"], dtype=np.float32)
        embs.append(vec)

    if not embs:
        raise ValueError("No embeddings found in jsonl")

    arr = np.stack(embs, axis=0).astype(np.float32, copy=False)  # [N, D]

    if l2_norm:
        arr = l2_normalize(arr)

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
