from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase, BatchEncoding

LABEL_MAP: Dict[str, int] = {
    "Wrong Number": 0, # Negative
    "Callback": 1, # Neutral/Intermediate
    "Promise to Pay": 2, # Positive
}
DATA_COLUMN = "transcription"
LABEL_COLUMN = "Disposition"

@dataclass
class ConversationExample:
    text: str
    label: int

class ConversationDataset(Dataset):
    def __init__(self, examples: Sequence[ConversationExample]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | int]:
        example = self.examples[idx]
        return {"text": example.text, "label": example.label}

def _prepare_examples(df: pd.DataFrame) -> List[ConversationExample]:
    examples = []
    for row in df.itertuples(index=False):
        text = getattr(row, DATA_COLUMN)
        label_raw = getattr(row, LABEL_COLUMN)
        if not isinstance(label_raw, str):
            label_raw = str(label_raw)
        label_key = label_raw.strip()
        examples.append(ConversationExample(text=text, label=LABEL_MAP[label_key]))
    return examples

def load_splits(csv_path: str, val_size: float = 0.1, test_size: float = 0.1, seed: int = 42):
    df = pd.read_csv(csv_path)
    examples = _prepare_examples(df)

    labels = [ex.label for ex in examples]
    texts = [ex.text for ex in examples]

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts,
        labels,
        test_size=val_size + test_size,
        random_state=seed,
        stratify=labels,
    )

    relative_val = val_size / (val_size + test_size)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=1 - relative_val,
        random_state=seed,
        stratify=temp_labels,
    )

    train_ds = ConversationDataset(
        [ConversationExample(text=t, label=l) for t, l in zip(train_texts, train_labels)]
    )
    val_ds = ConversationDataset(
        [ConversationExample(text=t, label=l) for t, l in zip(val_texts, val_labels)]
    )
    test_ds = ConversationDataset(
        [ConversationExample(text=t, label=l) for t, l in zip(test_texts, test_labels)]
    )

    classes = torch.tensor(sorted(set(LABEL_MAP.values())), dtype=torch.long)
    class_weights_np = compute_class_weight(
        class_weight="balanced", classes=classes.numpy(), y=train_labels
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float)

    return train_ds, val_ds, test_ds, class_weights

def build_collate_fn(tokenizer: PreTrainedTokenizerBase, max_length: int = 512):
    
    def collate(batch: Iterable[Dict[str, torch.Tensor | str | int]]) -> BatchEncoding:
        items = list(batch)
        texts = [str(item["text"]) for item in items]
        labels = torch.tensor([int(item["label"]) for item in items], dtype=torch.long)
        encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encodings["labels"] = labels
        return encodings

    return collate

def make_dataloaders(
    train_ds: ConversationDataset,
    val_ds: ConversationDataset,
    test_ds: ConversationDataset,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 8,
    max_length: int = 512,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    collate_fn = build_collate_fn(tokenizer=tokenizer, max_length=max_length)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return train_loader, val_loader, test_loader
