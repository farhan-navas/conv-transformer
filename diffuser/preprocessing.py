import re
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

# Tune these
SIM_THRESHOLD = 0.55          # below this => unknown
MARGIN_THRESHOLD = 0.03       # if p1 and p2 are too close => unknown
MAX_DUP_REPEAT = 2            # compress repeated short utterances

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

def normalize_text(s: str) -> str:
    s = s or ""
    s = s.replace("\u200b", " ").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def split_utterances(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    
    parts = SENT_SPLIT_RE.split(text)
    parts = [p.strip() for p in parts if p and p.strip()]
    return parts

def compress_repetitions(utterances: List[str]) -> List[str]:
    """Compress consecutive identical utterances and extreme token repeats."""
    out: List[str] = []
    last = None
    rep = 0

    for u in utterances:
        # Collapse extreme token repeats like "yes yes yes yes"
        toks = u.split()
        if len(toks) >= 6:
            # if >80% same token, compress to one token
            most = max(set(toks), key=toks.count)
            if toks.count(most) / len(toks) > 0.8:
                u = most

        if u == last:
            rep += 1
            if rep < MAX_DUP_REPEAT:
                out.append(u)
        else:
            last = u
            rep = 0
            out.append(u)
    return out

def cos_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T

def label_conversation(
    conversation: str,
    person1: str,
    person2: str,
    embedder: SentenceTransformer,
) -> List[Dict[str, Any]]:
    conv_utts = compress_repetitions(split_utterances(conversation))
    p1_utts = compress_repetitions(split_utterances(person1))
    p2_utts = compress_repetitions(split_utterances(person2))

    # Edge cases
    if not conv_utts:
        return []
    if not p1_utts and not p2_utts:
        return [{"speaker": "unknown", "text": u} for u in conv_utts] # "p1_sim": None, "p2_sim": None -> add for testing

    # Embed reference utterances
    p1_emb = embedder.encode(p1_utts, normalize_embeddings=True) if p1_utts else None
    p2_emb = embedder.encode(p2_utts, normalize_embeddings=True) if p2_utts else None

    # Embed conversation utterances
    conv_emb = embedder.encode(conv_utts, normalize_embeddings=True)

    labeled: List[Dict[str, Any]] = []

    for i, u in enumerate(conv_utts):
        ue = conv_emb[i:i+1]

        p1_best = -1.0
        p2_best = -1.0

        if p1_emb is not None and len(p1_emb) > 0:
            p1_best = float((ue @ p1_emb.T).max())

        if p2_emb is not None and len(p2_emb) > 0:
            p2_best = float((ue @ p2_emb.T).max())

        # Decide label
        speaker = "unknown"
        best = max(p1_best, p2_best)

        # If neither matches strongly, keep unknown
        if best >= SIM_THRESHOLD:
            # If they are too close, ambiguous -> unknown
            if abs(p1_best - p2_best) < MARGIN_THRESHOLD:
                speaker = "unknown"
            else:
                speaker = "person1" if p1_best > p2_best else "person2"

        labeled.append(
            {"speaker": speaker, "text": u} # "p1_sim": p1_best if p1_utts else None, "p2_sim": p2_best if p2_utts else None
        )

    return labeled

# merge consecutive utterances by the same speaker
def merge_consecutive(labeled: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not labeled:
        return labeled
    out = [dict(labeled[0])]
    for x in labeled[1:]:
        if x["speaker"] == out[-1]["speaker"]:
            out[-1]["text"] += " " + x["text"]
            # keep max similarity as a rough indicator
            # out[-1]["p1_sim"] = max(out[-1]["p1_sim"], x["p1_sim"]) if out[-1]["p1_sim"] is not None else x["p1_sim"]
            # out[-1]["p2_sim"] = max(out[-1]["p2_sim"], x["p2_sim"]) if out[-1]["p2_sim"] is not None else x["p2_sim"]
        else:
            out.append(dict(x))
    return out

def format_dialogue_to_str(merged: List[Dict[str, Any]]) -> str:
    lines = []
    for x in merged:
        sp = x["speaker"]
        tag = "person1" if sp == "person1" else ("person2" if sp == "person2" else "unknown")
        lines.append(f"{tag}: {x['text']}")
    return "\n".join(lines)

def format_dialogue_to_turns(merged):
    out = []
    for x in merged:
        sp = x["speaker"]
        tag = "person1" if sp == "person1" else ("person2" if sp == "person2" else "unknown")
        out.append({tag: x.get("text", "")})
    return out

if __name__ == "__main__":
    embedder = SentenceTransformer(MODEL_NAME)

    # JUST FOR EXAMPLE USAGE
    conversation, person1, person2 = "...", "...", "..."

    labeled = label_conversation(conversation, person1, person2, embedder)
    merged = merge_consecutive(labeled)
    print(format_dialogue_to_str(merged))
