import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import ensure_embeddings_npy
from dataset import EmbeddingDataset
from model import VQVAE

# ---- HARD-CODED PATHS (minimal) ----
JSONL_IN = "data/sentences.jsonl"
NPY_PATH = "data/embeddings.npy"
CKPT_PATH = "checkpoints/vqvae.pt"
JSONL_OUT = "data/sentences_with_cluster_id.jsonl"

BATCH_SIZE = 512
# -----------------------------------

def main():
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    dim_in = ckpt["dim_in"]
    latent_dim = ckpt["latent_dim"]
    num_codes = ckpt["num_codes"]
    hidden = ckpt["hidden"]
    beta = ckpt["beta"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VQVAE(dim_in, latent_dim, num_codes, hidden=hidden, beta=beta).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = EmbeddingDataset(NPY_PATH)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    all_ids = []
    with torch.no_grad():
        for x in dl:
            x = x.to(device)
            out = model(x)
            ids = out["indices"].detach().cpu().numpy().tolist()
            all_ids.extend(ids)

    # Write out JSONL with cluster_id added (same order)
    i = 0
    with open(JSONL_IN, "r", encoding="utf-8") as fin, open(JSONL_OUT, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["cluster_id"] = int(all_ids[i])
            i += 1
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[done] wrote {i} rows to {JSONL_OUT}")
    print(f"[codes] num_codes={num_codes}, used_unique={len(set(all_ids))}")

if __name__ == "__main__":
    main()
