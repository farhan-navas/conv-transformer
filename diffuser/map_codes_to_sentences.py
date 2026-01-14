import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

# Defaults (edit as needed)
CHECKPOINT_PATH = Path("checkpoints/vqvae.pt")
SENTENCE_EMBEDDINGS_PATH = Path("sentence_embeddings.jsonl")
CONVERSATION_PATH = Path("conversation_turns.jsonl")
OUT_PATH = Path("code_to_sentences.jsonl")
TOP_K = 5
BATCH_SIZE = 512


def load_sentence_vectors(
    sentence_embeddings_path: Path, conversation_path: Path
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Load sentence vectors and metadata aligned to conversation turns."""
    with conversation_path.open("r") as f:
        conv = json.load(f)
    with sentence_embeddings_path.open("r") as f:
        sent = json.load(f)

    conv_turns = conv.get("conversation_turns", [])
    sent_turns = sent.get("conversation_turns", [])
    if len(conv_turns) != len(sent_turns):
        raise ValueError(
            f"Turn count mismatch: conversation has {len(conv_turns)} turns, "
            f"sentence embeddings has {len(sent_turns)} turns"
        )

    vectors: List[np.ndarray] = []
    meta: List[Dict[str, Any]] = []
    for turn_idx, (conv_turn, sent_turn) in enumerate(zip(conv_turns, sent_turns)):
        # conv_turn is expected to be a dict with a single speaker key -> utterance text
        speaker_keys = list(conv_turn.keys())
        text = conv_turn.get(speaker_keys[0], "") if speaker_keys else ""
        speaker = sent_turn.get("speaker") or (speaker_keys[0] if speaker_keys else "")

        embedding_list = sent_turn.get("embeddings", [])
        for emb_idx, vec in enumerate(embedding_list):
            arr = np.asarray(vec, dtype=np.float32)
            vectors.append(arr)
            meta.append(
                {
                    "text": text,
                    "speaker": speaker,
                    "turn_index": turn_idx,
                    "embedding_index": emb_idx,
                }
            )

    if not vectors:
        raise ValueError("No sentence embeddings found")

    matrix = np.vstack(vectors)
    return matrix, meta


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


def build_mapping(
    checkpoint_path: Path,
    sentence_embeddings_path: Path,
    conversation_path: Path,
    out_path: Path,
    k: int,
    batch_size: int,
) -> None:
    sentences, meta = load_sentence_vectors(sentence_embeddings_path, conversation_path)
    codes = load_codebook_from_checkpoint(checkpoint_path)

    if codes.ndim != 2:
        raise ValueError(f"Expected code vectors with shape (n_codes, dim), got {codes.shape}")
    if sentences.shape[1] != codes.shape[1]:
        raise ValueError(
            f"Dim mismatch: code dim {codes.shape[1]} vs sentence dim {sentences.shape[1]}"
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
                        "turn_index": int(m["turn_index"]),
                        "embedding_index": int(m["embedding_index"]),
                        "speaker": m["speaker"],
                        "distance_l2": float(dist),
                        "text": m["text"],
                    }
                )
            f.write(json.dumps({"code_id": code_id, "neighbors": neighbors}) + "\n")

    print(f"Wrote {out_path} with top-{k} neighbors for {codes.shape[0]} code vectors")


def main() -> None:
    build_mapping(
        checkpoint_path=CHECKPOINT_PATH,
        sentence_embeddings_path=SENTENCE_EMBEDDINGS_PATH,
        conversation_path=CONVERSATION_PATH,
        out_path=OUT_PATH,
        k=TOP_K,
        batch_size=BATCH_SIZE,
    )


if __name__ == "__main__":
    main()
