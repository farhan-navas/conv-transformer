import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from models.vq_vae.model import VQVAE

CHECKPOINT_PATH = Path("checkpoints/vqvae.pt")
SENTENCE_EMBEDDINGS_NPY = Path("embeddings.npy")
SENTENCE_TEXT_JSONL = Path("conversation_text.jsonl")  # 1:1 with embeddings.npy
OUT_PATH = Path("code_to_sentences.jsonl")
TOP_K = 5
BATCH_SIZE = 512

def load_sentence_vectors(
    embeddings_npy_path: Path, sentence_text_jsonl: Path
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Load sentence vectors from .npy and texts from JSONL (1:1 order)."""
    vectors = np.load(embeddings_npy_path).astype(np.float32)

    texts: List[str] = []
    with sentence_text_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj.get("text", ""))

    if vectors.shape[0] != len(texts):
        raise ValueError(
            f"Length mismatch: embeddings {vectors.shape[0]} vs texts {len(texts)} "
            f"from {sentence_text_jsonl}"
        )

    meta = [{"text": t, "sentence_index": i} for i, t in enumerate(texts)]
    return vectors, meta

def load_sentence_meta(sentence_text_jsonl: Path) -> List[Dict[str, Any]]:
    """Load just the text metadata without pulling embeddings into memory."""
    texts: List[str] = []
    with sentence_text_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj.get("text", ""))

    return [{"text": t, "sentence_index": i} for i, t in enumerate(texts)]

def l2_topk_numpy(
    codes: np.ndarray, sentences: np.ndarray, k: int, batch_size: int = 512
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute L2 top-k neighbors using NumPy (batched)."""
    sent_norm = np.sum(sentences * sentences, axis=1)
    idx_out = np.empty((codes.shape[0], k), dtype=np.int64)
    dist_out = np.empty((codes.shape[0], k), dtype=np.float32)

    for start in range(0, codes.shape[0], batch_size):
        end = min(start + batch_size, codes.shape[0])
        chunk = codes[start:end]
        chunk_norm = np.sum(chunk * chunk, axis=1)[:, None]
        dist = chunk_norm + sent_norm[None, :] - 2.0 * np.matmul(chunk, sentences.T)

        part_idx = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
        part_dist = np.take_along_axis(dist, part_idx, axis=1)
        order = np.argsort(part_dist, axis=1)
        sorted_idx = np.take_along_axis(part_idx, order, axis=1)
        sorted_dist = np.take_along_axis(part_dist, order, axis=1)

        idx_out[start:end] = sorted_idx
        dist_out[start:end] = sorted_dist

    return dist_out, idx_out

def load_codebook_from_checkpoint(ckpt_path: Path) -> np.ndarray:
    """Load codebook weights from a saved VQ-VAE checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)

    for key in (
        "vq.codebook.weight",
        "module.vq.codebook.weight",  # in case of DataParallel
    ):
        if key in state:
            weights = state[key]
            break
    else:
        raise KeyError(
            "Could not find codebook weights in checkpoint; expected 'vq.codebook.weight'"
        )

    if weights.ndim != 2:
        raise ValueError(f"Expected codebook weights of shape [n_codes, dim], got {weights.shape}")

    return weights.cpu().numpy().astype(np.float32)

def load_encoder_from_checkpoint(ckpt_path: Path) -> Tuple[VQVAE, Dict[str, Any]]:
    """Reconstruct the trained VQ-VAE to encode sentence embeddings to the codebook latent space."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = {
        "dim_in": int(ckpt.get("dim_in", 0)),
        "latent_dim": int(ckpt.get("latent_dim", 0)),
        "num_codes": int(ckpt.get("num_codes", 0)),
        "hidden": int(ckpt.get("hidden", 256)),
        "beta": float(ckpt.get("beta", 0.25)),
    }

    missing = [k for k, v in meta.items() if v == 0]
    if missing:
        raise ValueError(f"Checkpoint {ckpt_path} is missing required metadata: {missing}")

    model = VQVAE(
        dim_in=meta["dim_in"],
        dim_latent=meta["latent_dim"],
        num_codes=meta["num_codes"],
        hidden=meta["hidden"],
        beta=meta["beta"],
    )

    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model, meta

def encode_sentences_with_vqvae(
    embeddings_npy_path: Path,
    ckpt_path: Path,
    batch_size: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Project raw sentence embeddings into the VQ-VAE latent space (pre-quantization)."""
    model, meta = load_encoder_from_checkpoint(ckpt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    embs = np.load(embeddings_npy_path).astype(np.float32, copy=False)

    latents: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, embs.shape[0], batch_size):
            end = min(start + batch_size, embs.shape[0])
            batch = torch.from_numpy(embs[start:end]).to(device)
            z_e = model.encoder(batch)
            latents.append(z_e.cpu().numpy())

    return np.concatenate(latents, axis=0), meta

def build_mapping(
    checkpoint_path: Path,
    sentence_embeddings_npy: Path,
    sentence_text_jsonl: Path,
    out_path: Path,
    k: int,
    batch_size: int,
) -> None:
    meta = load_sentence_meta(sentence_text_jsonl)
    sentences, ckpt_meta = encode_sentences_with_vqvae(sentence_embeddings_npy, checkpoint_path, batch_size)
    codes = load_codebook_from_checkpoint(checkpoint_path)

    if sentences.shape[0] != len(meta):
        raise ValueError(
            f"Length mismatch after encoding: latents {sentences.shape[0]} vs texts {len(meta)} from {sentence_text_jsonl}"
        )

    if codes.ndim != 2:
        raise ValueError(f"Expected code vectors with shape (n_codes, dim), got {codes.shape}")
    if sentences.shape[1] != codes.shape[1]:
        raise ValueError(
            f"Dim mismatch after VQ-VAE encoding: code dim {codes.shape[1]} vs latent dim {sentences.shape[1]}"
        )

    distances, indices = l2_topk_numpy(codes, sentences, k, batch_size)

    with out_path.open("w") as f:
        for code_id, (nbr_idx, nbr_dist) in enumerate(zip(indices, distances)):
            neighbors: List[Dict[str, Any]] = []
            for rank, (sid, dist) in enumerate(zip(nbr_idx.tolist(), nbr_dist.tolist())):
                m = meta[sid]
                neighbors.append(
                    {
                        "rank": rank + 1,
                        "sentence_index": int(sid),
                        "distance_l2": float(dist),
                        "text": m["text"],
                    }
                )
            f.write(json.dumps({"code_id": code_id, "neighbors": neighbors}) + "\n")

    print(f"Wrote {out_path} with top-{k} neighbors for {codes.shape[0]} code vectors")

def main() -> None:
    build_mapping(
        checkpoint_path=CHECKPOINT_PATH,
        sentence_embeddings_npy=SENTENCE_EMBEDDINGS_NPY,
        sentence_text_jsonl=SENTENCE_TEXT_JSONL,
        out_path=OUT_PATH,
        k=TOP_K,
        batch_size=BATCH_SIZE,
    )

if __name__ == "__main__":
    main()
