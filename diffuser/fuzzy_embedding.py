import json
from typing import List, Dict, Any

import pandas as pd
from sentence_transformers import SentenceTransformer
from preprocessing import (
    label_conversation,
    merge_consecutive,
    format_dialogue_to_turns,
    split_utterances,
)

INPUT_CSV = "data.csv"
CONVERSATIONS_JSONL = "conversation_turns.jsonl"
EMBEDDINGS_JSONL = "sentence_embeddings.jsonl"

TARGET_COLUMNS = ["transcription", "transcription_ch0", "transcription_ch1"]
MODEL_NAME = "all-MiniLM-L6-v2"

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_row_items(df, embedder: SentenceTransformer) -> List[Dict[str, Any]]:
    items = []
    for row_idx, row in df.iterrows():
        conv = str(row.get("transcription", "") or "")
        p1 = str(row.get("transcription_ch0", "") or "")
        p2 = str(row.get("transcription_ch1", "") or "")

        labeled = label_conversation(conv, p1, p2, embedder)
        merged = merge_consecutive(labeled)
        formatted = format_dialogue_to_turns(merged)
        items.append({"row_idx": int(row_idx), "conversation_turns": formatted})
        break

    write_jsonl(CONVERSATIONS_JSONL, items)
    return items


def build_sentence_embeddings(
    row_items: List[Dict[str, Any]], embedder: SentenceTransformer
) -> List[Dict[str, Any]]:
    """Create per-sentence embeddings grouped by speaker for each row."""
    emb_rows: List[Dict[str, Any]] = []

    for item in row_items:
        embeddings: Dict[str, List[List[float]]] = {}
        for turn in item.get("conversation_turns", []):
            for speaker, text in turn.items():
                sentences = split_utterances(str(text or ""))
                if not sentences:
                    continue

                sent_emb = embedder.encode(sentences, normalize_embeddings=True)
                sent_emb_list = sent_emb.tolist() if hasattr(sent_emb, "tolist") else []
                if not sent_emb_list:
                    continue

                if speaker not in embeddings:
                    embeddings[speaker] = []
                embeddings[speaker].extend(sent_emb_list)

        emb_rows.append({"row_idx": item["row_idx"], "embeddings": embeddings})
    
    return emb_rows

def main():
    df = pd.read_csv(INPUT_CSV)

    embedder = SentenceTransformer(MODEL_NAME)
    row_items = build_row_items(df, embedder)

    sentence_embeddings = build_sentence_embeddings(row_items, embedder)
    write_jsonl(EMBEDDINGS_JSONL, sentence_embeddings)

    print(f"Done. Wrote {len(sentence_embeddings)} rows to {EMBEDDINGS_JSONL}")
    print(f"Model: {MODEL_NAME}")

if __name__ == "__main__":
    main()
