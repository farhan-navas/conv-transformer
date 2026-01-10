import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import EmbeddingDataset
from .model import VQVAE


@dataclass
class ExportCodesConfig:
    jsonl_in: str = "sentence_embeddings.jsonl"
    npy_path: str = "embeddings.npy"
    checkpoint_path: str = "checkpoints/vqvae.pt"
    jsonl_out: str = "sentence_cluster_ids.jsonl"
    batch_size: int = 512


def export_codes(config: ExportCodesConfig) -> None:
    ckpt = torch.load(config.checkpoint_path, map_location="cpu")
    dim_in = ckpt["dim_in"]
    latent_dim = ckpt["latent_dim"]
    num_codes = ckpt["num_codes"]
    hidden = ckpt["hidden"]
    beta = ckpt["beta"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VQVAE(dim_in, latent_dim, num_codes, hidden=hidden, beta=beta).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = EmbeddingDataset(config.npy_path)
    dl = DataLoader(ds, batch_size=config.batch_size, shuffle=False)

    all_ids = []
    with torch.no_grad():
        for x in dl:
            x = x.to(device)
            out = model(x)
            ids = out["indices"].detach().cpu().numpy().tolist()
            all_ids.extend(ids)

    idx = 0
    written = 0
    out_path = Path(config.jsonl_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config.jsonl_in, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
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

    print(f"[done] wrote {written} rows to {out_path}")
    print(f"[codes] num_codes={num_codes}, used_unique={len(set(all_ids))}")

