import json
from math import sqrt
from typing import List, Dict, Any

import pandas as pd
from sentence_transformers import SentenceTransformer
from data_prep import label_conversation, merge_consecutive, format_dialogue_to_turns, split_utterances

INPUT_CSV = "data-human.csv"
CONVERSATIONS_JSONL = "conversation_turns.jsonl"
EMBEDDINGS_JSONL = "sentence_embeddings.jsonl"
METRICS_JSON = "overall_metrics.json"

# TARGET_COLUMNS = ["transcription", "transcription_ch0", "transcription_ch1"]
MODEL_NAME = "all-MiniLM-L6-v2"

# New stacked input schema (one row per channel)
STACKED_COLS = [
    "transcription", "channel_text", "channel", "Disposition", "orig_idx",
]

def write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _aggregate_stacked(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Rebuild per-conversation records from stacked channel rows."""
    missing = [c for c in STACKED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    grouped = df.groupby("orig_idx", sort=False)
    records: List[Dict[str, Any]] = []
    for gid, g in grouped:
        conv = str(g["transcription"].iloc[0] or "")
        disp = str(g["Disposition"].iloc[0] or "")
        ch0 = g.loc[g["channel"] == "ch0", "channel_text"].astype(str)
        ch1 = g.loc[g["channel"] == "ch1", "channel_text"].astype(str)
        p1 = ch0.iloc[0] if len(ch0) else ""
        p2 = ch1.iloc[0] if len(ch1) else ""

        if not conv and not p1 and not p2:
            continue

        len_conv = len(conv.split())
        if len_conv == 0:
            len_conv = len((p1 + " " + p2).split())

        records.append({
            "orig_idx": gid,
            "transcription": conv,
            "p1": p1,
            "p2": p2,
            "disposition": disp,
            "word_count": len_conv,
        })
    return records


def _map_outcome(disp: str) -> int:
    if disp == "Promise to Pay":
        return 1
    if disp == "Callback":
        return 0
    return -1

def build_row_items(df, embedder: SentenceTransformer) -> List[Dict[str, Any]]:
    print("[build_row_items] start")
    items = []
    # If dataset is stacked (channel per row), rebuild conversations first
    if "channel" in df.columns and "channel_text" in df.columns:
        rows = _aggregate_stacked(df)
    else:
        # fallback to old schema
        rows = [{
            "orig_idx": idx,
            "transcription": str(row.get("transcription", "") or ""),
            "p1": str(row.get("transcription_ch0", "") or ""),
            "p2": str(row.get("transcription_ch1", "") or ""),
            "disposition": str(row.get("Disposition", "")),
            "word_count": len(str(row.get("transcription", "") or "").split()),
        } for idx, row in df.iterrows()]

    for rec in rows:
        conv = rec["transcription"]
        p1 = rec["p1"]
        p2 = rec["p2"]
        len_conv = rec["word_count"]
        outcome = _map_outcome(rec.get("disposition", ""))

        labeled = label_conversation(conv, p1, p2, embedder)
        merged = merge_consecutive(labeled)
        merged = [m for m in merged if m.get("speaker") != "unknown"]
        formatted = format_dialogue_to_turns(merged)
        items.append({
            "row_idx": int(rec["orig_idx"]),
            "conversation_turns": formatted,
            "number_of_turns": len(formatted),
            "word_count": len_conv,
            "outcome": outcome,
        })

        if len(items) % 100 == 0:
            print(f"[build_row_items] processed row {len(items)}")

    write_jsonl(CONVERSATIONS_JSONL, items)
    print(f"[build_row_items] wrote {len(items)} rows to {CONVERSATIONS_JSONL}")
    return items

def build_sentence_embeddings(row_items: List[Dict[str, Any]], embedder: SentenceTransformer):
    print("[build_sentence_embeddings] start")
    emb_rows: List[Dict[str, Any]] = []

    for item in row_items:
        turn_embs: List[Dict[str, Any]] = []

        for turn in item.get("conversation_turns", []):
            if not turn:
                continue

            speaker, text = next(iter(turn.items()))
            sentences = split_utterances(str(text or ""))
            sent_emb_list: List[List[float]] = []

            if sentences:
                sent_emb = embedder.encode(sentences, normalize_embeddings=True)
                sent_emb_list = sent_emb.tolist() if hasattr(sent_emb, "tolist") else []

            turn_embs.append({
                "speaker": speaker,
                "embeddings": sent_emb_list,
            })

        emb_rows.append({
            "row_idx": item["row_idx"],
            "conversation_turns": turn_embs,
        })

        if len(emb_rows) % 100 == 0:
            print(f"[build_sentence_embeddings] processed {len(emb_rows)} rows")
    
    print(f"[build_sentence_embeddings] built embeddings for {len(emb_rows)} rows")
    return emb_rows

def _stats(vals: List[int]) -> Dict[str, Any]:
    n = len(vals)
    s = sum(vals)
    avg = s / n
    variance = sum((v - avg) ** 2 for v in vals) / n

    return {
        "min": min(vals),
        "max": max(vals),
        "avg": avg,
        "std": sqrt(variance),
    }

def build_overall_metrics(row_items: List[Dict[str, Any]]):
    print("[build_overall_metrics] start")
    cats = {
        1: {"label": "positive", "nts": [], "wcs": []},
        0: {"label": "intermediate", "nts": [], "wcs": []},
        -1: {"label": "negative", "nts": [], "wcs": []},
    }

    for item in row_items:
        outcome = item.get("outcome")
        if outcome not in cats:
            continue
        cats[outcome]["nts"].append(int(item.get("number_of_turns", 0)))
        cats[outcome]["wcs"].append(int(item.get("word_count", 0)))

    metrics: Dict[str, Any] = {}
    for _, cat in cats.items():
        metrics[cat["label"]] = {
            "count": len(cat["nts"]),
            "number_of_turns": _stats(cat["nts"]),
            "word_count": _stats(cat["wcs"]),
        }

    write_json(METRICS_JSON, metrics)
    print(f"[build_overall_metrics] wrote metrics to {METRICS_JSON}")

def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"[main] loaded {len(df)} rows from {INPUT_CSV}")

    embedder = SentenceTransformer(MODEL_NAME)
    print(f"[main] loaded model {MODEL_NAME}")
    row_items = build_row_items(df, embedder)

    sentence_embeddings = build_sentence_embeddings(row_items, embedder)
    write_jsonl(EMBEDDINGS_JSONL, sentence_embeddings)
    build_overall_metrics(row_items)

    print(f"Done. Wrote {len(sentence_embeddings)} rows to {EMBEDDINGS_JSONL}")
    print(f"Wrote metrics to {METRICS_JSON}")
    print(f"Model: {MODEL_NAME}")

if __name__ == "__main__":
    main()
