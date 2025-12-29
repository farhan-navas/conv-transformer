import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import EmbeddingDataset
from model import VQVAE

# change this later, put in data/
JSONL_IN = "sentence_embeddings.jsonl"
NPY_PATH = "embeddings.npy"
CKPT_PATH = "checkpoints/vqvae.pt"
JSONL_OUT = "sentence_cluster_ids.jsonl"

BATCH_SIZE = 512

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

    # Write out JSONL with cluster_ids aggregated per speaker per original row
    idx = 0
    written = 0
    with open(JSONL_IN, "r", encoding="utf-8") as fin, open(JSONL_OUT, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            speakers: dict = {}
            turns = obj.get("conversation_turns", [])
            for turn in turns:
                if not turn:
                    continue
                speaker = turn.get("speaker") or next(iter(turn.keys()))
                emb_list = turn.get("embeddings", [])
                count = len(emb_list)
                if count == 0:
                    continue
                # slice cluster ids for this turn
                turn_ids = all_ids[idx: idx + count]
                idx += count
                speakers.setdefault(speaker, []).extend(int(cid) for cid in turn_ids)

            out_row = {
                "row_idx": obj.get("row_idx"),
                "speakers": [
                    {"speaker": spk, "cluster_ids": ids}
                    for spk, ids in speakers.items()
                ],
            }
            written += 1
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    if idx != len(all_ids):
        print(f"[warn] consumed {idx} cluster ids, but {len(all_ids)} were computed; some ids may be unused")

    print(f"[done] wrote {written} rows to {JSONL_OUT}")
    print(f"[codes] num_codes={num_codes}, used_unique={len(set(all_ids))}")

if __name__ == "__main__":
    main()
