import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase, BatchEncoding

@dataclass
class DataConfig:
    csv_path: str = "data.csv"
    text_column: str = "transcription"
    label_column: str = "Disposition"
    label_map: Dict[str, int] = field(
        default_factory=lambda: {
            "Wrong Number": 0,
            "Callback": 1,
            "Promise to Pay": 2,
        }
    )
    val_size: float = 0.1
    test_size: float = 0.1
    seed: int = 42

@dataclass
class ConversationExample:
    text: str
    label: int


@dataclass
class CodeExample:
    codes: List[int]
    label: int

class ConversationDataset(Dataset):
    def __init__(self, examples: Sequence[ConversationExample]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | int]:
        example = self.examples[idx]
        return {"text": example.text, "label": example.label}


class CodeDataset(Dataset):
    def __init__(self, examples: Sequence[CodeExample]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | List[int] | int]:
        example = self.examples[idx]
        return {"codes": example.codes, "label": example.label}

def _prepare_examples(df: pd.DataFrame, data_cfg: DataConfig) -> List[ConversationExample]:
    examples: List[ConversationExample] = []
    for row in df.itertuples(index=False):
        text = getattr(row, data_cfg.text_column)
        label_raw = getattr(row, data_cfg.label_column)
        if not isinstance(label_raw, str):
            label_raw = str(label_raw)
        label_key = label_raw.strip()
        examples.append(ConversationExample(text=text, label=data_cfg.label_map[label_key]))
    return examples

def load_splits(csv_path: str, data_cfg: DataConfig) -> Tuple[ConversationDataset, ConversationDataset, ConversationDataset, torch.Tensor]:
    # Keep column names and label mapping configurable to support new datasets.
    df = pd.read_csv(csv_path)
    examples = _prepare_examples(df, data_cfg)

    labels = [ex.label for ex in examples]
    texts = [ex.text for ex in examples]

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts,
        labels,
        test_size=data_cfg.val_size + data_cfg.test_size,
        random_state=data_cfg.seed,
        stratify=labels,
    )

    relative_val = data_cfg.val_size / (data_cfg.val_size + data_cfg.test_size)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=1 - relative_val,
        random_state=data_cfg.seed,
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

    classes = torch.tensor(sorted(set(data_cfg.label_map.values())), dtype=torch.long)
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


def _load_labels(data_cfg: DataConfig) -> Dict[int, int]:
    df = pd.read_csv(data_cfg.csv_path)
    labels: Dict[int, int] = {}
    for idx, row in df.iterrows():
        label_raw = row.get(data_cfg.label_column, "")
        label_key = str(label_raw).strip()
        labels[idx] = data_cfg.label_map[label_key]
    return labels


def _load_code_examples(codes_jsonl: str, data_cfg: DataConfig) -> List[CodeExample]:
    labels = _load_labels(data_cfg)
    examples: List[CodeExample] = []
    with open(codes_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            row_idx = int(obj.get("row_idx"))
            if row_idx not in labels:
                continue
            seq: List[int] = []
            for speaker in obj.get("speakers", []):
                seq.extend(int(cid) for cid in speaker.get("cluster_ids", []))
            examples.append(CodeExample(codes=seq, label=labels[row_idx]))
    return examples


def load_code_splits(codes_jsonl: str, data_cfg: DataConfig) -> Tuple[CodeDataset, CodeDataset, CodeDataset, torch.Tensor]:
    examples = _load_code_examples(codes_jsonl, data_cfg)

    labels = [ex.label for ex in examples]
    codes_all = [ex.codes for ex in examples]

    train_codes, temp_codes, train_labels, temp_labels = train_test_split(
        codes_all,
        labels,
        test_size=data_cfg.val_size + data_cfg.test_size,
        random_state=data_cfg.seed,
        stratify=labels,
    )

    relative_val = data_cfg.val_size / (data_cfg.val_size + data_cfg.test_size)
    val_codes, test_codes, val_labels, test_labels = train_test_split(
        temp_codes,
        temp_labels,
        test_size=1 - relative_val,
        random_state=data_cfg.seed,
        stratify=temp_labels,
    )

    train_ds = CodeDataset([CodeExample(codes=c, label=l) for c, l in zip(train_codes, train_labels)])
    val_ds = CodeDataset([CodeExample(codes=c, label=l) for c, l in zip(val_codes, val_labels)])
    test_ds = CodeDataset([CodeExample(codes=c, label=l) for c, l in zip(test_codes, test_labels)])

    classes = torch.tensor(sorted(set(data_cfg.label_map.values())), dtype=torch.long)
    class_weights_np = compute_class_weight(
        class_weight="balanced", classes=classes.numpy(), y=train_labels
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float)

    return train_ds, val_ds, test_ds, class_weights


def build_code_collate_fn(max_length: int, pad_id: int, cls_id: int | None = None):

    def collate(batch: Iterable[Dict[str, torch.Tensor | List[int] | int]]) -> Dict[str, torch.Tensor]:
        items = list(batch)
        labels = torch.tensor([int(item["label"]) for item in items], dtype=torch.long)
        seqs: List[List[int]] = []

        for item in items:
            codes = list(item.get("codes", []))
            if cls_id is not None:
                codes = [cls_id] + codes
            if len(codes) > max_length:
                codes = codes[-max_length:]
            seqs.append(codes)

        padded: List[List[int]] = []
        masks: List[List[int]] = []
        for seq in seqs:
            pad_len = max_length - len(seq)
            padded_seq = seq + [pad_id] * pad_len
            mask = [1] * len(seq) + [0] * pad_len
            padded.append(padded_seq)
            masks.append(mask)

        input_ids = torch.tensor(padded, dtype=torch.long)
        attention_mask = torch.tensor(masks, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate


def make_code_dataloaders(
    train_ds: CodeDataset,
    val_ds: CodeDataset,
    test_ds: CodeDataset,
    batch_size: int,
    max_length: int,
    pad_id: int,
    cls_id: int | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    collate_fn = build_code_collate_fn(max_length=max_length, pad_id=pad_id, cls_id=cls_id)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader
