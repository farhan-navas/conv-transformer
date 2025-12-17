import json
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer
from preprocessing import load_csv, build_row_items

INPUT_CSV = "data.csv"
OUTPUT_JSONL = "sentence_embeddings.jsonl"

TARGET_COLUMNS = ["transcription", "transcription_ch0", "transcription_ch1"]
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    df = load_csv(INPUT_CSV)

    cols_present = [c for c in TARGET_COLUMNS if c in df.columns]
    if not cols_present:
        raise SystemExit(f"No target columns, only: {list(df.columns)}")

    row_items = build_row_items(df, TARGET_COLUMNS)
    embedder = SentenceTransformer(MODEL_NAME)

    # Flatten all sentences across all rows+cols so we can embed in batches
    flat_sentences: List[str] = []
    flat_index: List[tuple[int, str, int]] = []  # (row_item_idx, col, sent_idx)

    for ri, item in enumerate(row_items):
        for col in cols_present:
            sents = item["sentences"].get(col, [])
            for si, sent in enumerate(sents):
                flat_sentences.append(sent)
                flat_index.append((ri, col, si))

    out_rows: List[Dict[str, Any]] = []
    for item in row_items:
        out_rows.append(
            {
                "row_idx": item["row_idx"],
                "sentences": item["sentences"],
                "embeddings": {c: [] for c in cols_present},
            }
        )

    # Embed and place back into per-row structure
    if flat_sentences:
        embeddings = embedder.encode(
            flat_sentences,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        # Create slots of correct length per (row, col)
        for ri, item in enumerate(row_items):
            for col in cols_present:
                n = len(item["sentences"].get(col, []))
                out_rows[ri]["embeddings"][col] = [None] * n

        # Fill slots
        for (ri, col, si), emb in zip(flat_index, embeddings):
            out_rows[ri]["embeddings"][col][si] = emb.tolist()

    write_jsonl(OUTPUT_JSONL, out_rows)

    print(f"Done. Wrote {len(out_rows)} row objects to {OUTPUT_JSONL}")
    print(f"Columns processed: {cols_present}")
    print(f"Model: {MODEL_NAME}")

if __name__ == "__main__":
    main()
