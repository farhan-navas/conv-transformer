from typing import List, Dict, Any

import re
import pandas as pd

# Keep the same regex splitter you used
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

def split_into_sentences(text: str):
    text = (text or "").strip()
    if not text:
        return []
    parts = SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]

def build_row_items(
    df: pd.DataFrame,
    target_columns: List[str],
) -> List[Dict[str, Any]]:
    cols = [c for c in target_columns if c in df.columns]

    if not cols:
        raise ValueError(f"Don't have transcription columns, columns present: {list(df.columns)}")

    items = []
    for row_idx, row in df.iterrows():
        sentences_by_col: Dict[str, List[str]] = {}

        for col in cols:
            val = row.get(col)
            if pd.isna(val):
                sentences_by_col[col] = []
                continue

            text = str(val)
            sentences_by_col[col] = split_into_sentences(text)

        items.append(
            {
                "row_idx": row_idx,
                "sentences": sentences_by_col,
            }
        )

    return items

def load_csv(input_csv: str) -> pd.DataFrame:
    return pd.read_csv(input_csv)
