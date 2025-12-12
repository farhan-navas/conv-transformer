from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase, BatchEncoding

DEFAULT_LABEL_MAP: Dict[str, int] = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

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


def _prepare_examples(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    label_map: Dict[str, int],
) -> List[ConversationExample]:
    if text_column not in df.columns:
        raise ValueError(f"Missing text column '{text_column}' in data")
    if label_column not in df.columns:
        raise ValueError(f"Missing label column '{label_column}' in data")

    examples: List[ConversationExample] = []
    for row in df.itertuples(index=False):
        text = getattr(row, text_column)
        label_raw = getattr(row, label_column)
        if not isinstance(label_raw, str):
            label_raw = str(label_raw)
        label_key = label_raw.strip().lower()
        if label_key not in label_map:
            raise ValueError(f"Unknown label '{label_raw}' encountered; expected one of {list(label_map)}")
        examples.append(ConversationExample(text=text, label=label_map[label_key]))
    return examples


def load_splits(
    csv_path: str,
    text_column: str = "transcript",
    label_column: str = "outcome",
    label_map: Dict[str, int] | None = None,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
) -> Tuple[ConversationDataset, ConversationDataset, ConversationDataset, torch.Tensor]:
    label_map = label_map or DEFAULT_LABEL_MAP
    df = pd.read_csv(csv_path)
    examples = _prepare_examples(df, text_column, label_column, label_map)

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

    classes = torch.tensor(sorted(set(label_map.values())), dtype=torch.long)
    class_weights_np = compute_class_weight(
        class_weight="balanced", classes=classes.numpy(), y=train_labels
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float)

    return train_ds, val_ds, test_ds, class_weights


def build_collate_fn(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
) -> Callable[[Iterable[Dict[str, torch.Tensor | str | int]]], BatchEncoding]:
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
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    collate_fn = build_collate_fn(tokenizer=tokenizer, max_length=max_length)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn
    )
    return train_loader, val_loader, test_loader
