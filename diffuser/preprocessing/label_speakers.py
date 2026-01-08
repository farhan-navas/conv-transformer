import csv
import os, re
from typing import List
import numpy as np
import pandas as pd
import joblib
from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm

INPUT_CSV = "data.csv"
OUTPUT_CSV = "data-human.csv"

LABEL_MAP = {0: "Agent", 1: "Donor"}

RE_SPACES        = re.compile(r"\s+")
RE_URL           = re.compile(r"https?://\S+|www\.\S+")
RE_EMAIL         = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
RE_LEADING_CSV   = re.compile(r"^\s*(?:\d+(?:[.,]\d+)?\s*,\s*){2,}")

RE_CURRENCY = re.compile(r"(?:₹|rs\.?|inr|rupees?)\s*\d[\d,.\-]*", re.IGNORECASE)
RE_NUM      = re.compile(r"(?<![a-z])\d[\d,./-]*")

RE_PHONE = re.compile(r"""
    (?<!\w)                                  # no letter/digit before
    (?:\+?\d{1,3}[\s\-\(\)]*)?               # optional country code
    (?:\(?\d{2,4}\)?[\s\-\(\)]*){2,3}        # middle groups
    \d{3,4}                                  # final group
    (?!\w)                                   # no letter/digit after
""", re.VERBOSE)

DESIRED_COLS = [
    "call_date", "entity_details_id", "call_duration", "call_response",
    "audio_file_path", "Caller", "Disposition", "transcription",
    "channel_text", "channel", 'channel_clean'
]

def build_stacked(df: pd.DataFrame) -> pd.DataFrame:
    # preserve original order without leaving helper cols behind
    df2 = df.reset_index(drop=False).rename(columns={"index": "_row"})

    ch0 = (
        df2.drop(columns=["transcription_ch1"])
           .rename(columns={"transcription_ch0": "channel_text"})
           .assign(channel="ch0", _chan_ord=0)
    )
    ch1 = (
        df2.drop(columns=["transcription_ch0"])
           .rename(columns={"transcription_ch1": "channel_text"})
           .assign(channel="ch1", _chan_ord=1)
    )

    stacked = (
        pd.concat([ch0, ch1], ignore_index=True)
          .sort_values(["_row", "_chan_ord"], kind="stable")
          .drop(columns=["_row", "_chan_ord"])
          .reset_index(drop=True)
    )

    # tidy text
    stacked["channel_text"] = stacked["channel_text"].fillna("").astype(str).str.strip()
    return stacked

def _cap_consecutive_tokens(tokens: List[str], max_repeat: int = 3) -> List[str]:
    out, prev, cnt = [], None, 0
    for t in tokens:
        if t == prev:
            cnt += 1
        else:
            prev, cnt = t, 1
        if cnt <= max_repeat:
            out.append(t)
    return out

def _cap_repeated_phrases_any(tokens: List[str], min_n=2, max_n=7, max_repeat=3) -> List[str]:
    if len(tokens) < min_n * 2:
        return tokens
    out = tokens
    for n in range(min_n, max_n + 1):
        i, new_out = 0, []
        while i < len(out):
            win = out[i:i+n]
            j, rep = i + n, 1
            while j + n <= len(out) and out[j:j+n] == win:
                rep += 1
                j += n
            new_out.extend(win * min(rep, max_repeat))
            i = j
        out = new_out
    return out

def clean_text(
    s: str,
    normalize_numbers: bool = True,
    cap_token_repeat: int = 3,
    cap_phrase_any: bool = True,
    phrase_min_n: int = 2,
    phrase_max_n: int = 7,
    phrase_max_repeat: int = 3
) -> str:
    if not isinstance(s, str):
        return ""

    s = s.replace("\u2019", "'").replace("\u2018", "'").lower().strip()

    s = RE_LEADING_CSV.sub("", s)

    s = RE_URL.sub(" <URL> ", s)
    s = RE_EMAIL.sub(" <EMAIL> ", s)

    def _replace_phone(m: re.Match) -> str:
        span = m.group(0)
        digits = re.sub(r"\D", "", span)
        if 10 <= len(digits) <= 15:
            return " <phone> "
        return span
    s = RE_PHONE.sub(_replace_phone, s)

    if normalize_numbers:
        s = RE_CURRENCY.sub(" <cur> <num> ", s)
        s = RE_NUM.sub(" <num> ", s)

    s = re.sub(r"[^a-zA-Z<>' ]", " ", s)

    s = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", s)

    s = RE_SPACES.sub(" ", s).strip()
    if not s:
        return ""

    toks = s.split()
    if cap_token_repeat is not None:
        toks = _cap_consecutive_tokens(toks, max_repeat=cap_token_repeat)
    if cap_phrase_any:
        toks = _cap_repeated_phrases_any(
            toks, min_n=phrase_min_n, max_n=phrase_max_n, max_repeat=phrase_max_repeat
        )
    s = " ".join(toks)

    s = (
        s.replace("<num>", "<NUM>")
         .replace("<cur>", "<CUR>")
         .replace("<phone>", "<PHONE>")
         .replace("<url>", "<URL>")
         .replace("<email>", "<EMAIL>")
    )
    return s

def prepare_text(df: pd.DataFrame, text_col: str = "channel_text") -> pd.DataFrame:
    """
    Clean the specified text_col and KEEP all requested metadata columns.

    Returns a DataFrame containing:
      - all columns in DESIRED_COLS that exist in df
      - 'orig_idx' (source row index)
      - 'text_clean' (cleaned version of text_col)
    """
    # Figure out which of the desired columns actually exist
    present_cols = [c for c in DESIRED_COLS if c in df.columns]
    if text_col not in df.columns:
        raise ValueError(f"[ERROR] text_col '{text_col}' not found in input DataFrame.")

    keep_cols = list(dict.fromkeys(present_cols + [text_col]))  # de-dup while preserving order
    D = df.loc[:, keep_cols].copy()

    # D = D.dropna(subset=[text_col])  # remove NaN/empty text rows
    D["orig_idx"] = D.index

    texts = D[text_col].astype(str).tolist()

    n_threads = min(32, os.cpu_count() or 8)
    print(f"[INFO] Cleaning {len(texts):,} texts from '{text_col}' using {n_threads} threads...")
    with ThreadPool(n_threads) as pool:
        D["text_clean"] = list(
            tqdm(pool.imap(clean_text, texts, chunksize=500),
                 total=len(texts), desc="Cleaning")
        )

    before = len(D)
    # D = D[D["text_clean"].str.strip() != ""].reset_index(drop=True)
    print(f"[INFO] Cleaning complete. Retained {len(D):,}/{before:,} rows (no post-filter).")

    final_cols = present_cols + ["orig_idx", "text_clean"]
    D = D.loc[:, final_cols]

    return D

def predict_role_batch(texts, role_model):
    """Return predictions + confidences for a list of texts."""
    texts = [str(t or "").strip() for t in texts]
    preds = role_model.predict(texts)
    scores = role_model.decision_function(texts)

    if scores.ndim == 1:
        confs = 1 / (1 + np.exp(-np.abs(scores)))  # squash distance into 0..1
    else:
        max_scores = np.max(scores, axis=1)
        confs = 1 / (1 + np.exp(-np.abs(max_scores)))

    labels = [LABEL_MAP[p] for p in preds]
    return preds, labels, confs

def main():
    dataset = pd.read_csv(INPUT_CSV)

    df_stacked = build_stacked(dataset)
    df_stacked_clean = prepare_text(df_stacked, text_col="channel_text")

    role_model = joblib.load("diffuser/preprocessing/CLEAN_agent_donor.joblib")
    df_human = df_stacked_clean.copy()
    preds, labels, confs = predict_role_batch(df_human["text_clean"].tolist(), role_model)

    df_human["role_label"] = labels
    df_human["role_confidence"] = np.round(confs, 3)

    df_human.to_csv(OUTPUT_CSV, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

if __name__ == "__main__":
    main()
