from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .data import ConversationDataset, build_collate_fn, load_splits, make_dataloaders

@dataclass
class TrainConfig:
    csv_path: str
    model_name: str = "roberta-base"
    batch_size: int = 8
    max_length: int = 512
    epochs: int = 3
    learning_rate: float = 2e-5
    seed: int = 42
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    log_path: str = "metrics.jsonl"

def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _log_metrics(log_path: str, record: Dict) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record))
        f.write("\n")

def _evaluate(model: torch.nn.Module, data_loader: DataLoader, device: torch.device):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in data_loader:
            batch = _to_device(batch, device)
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            all_labels.extend(batch["labels"].cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    per_class = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    per_class_f1 = {k: v.get("f1-score", 0.0) for k, v in per_class.items() if k.isdigit()} # type: ignore
    return accuracy, macro_f1, per_class_f1

def train_model(config: TrainConfig) -> Dict[str, float]:
    torch.manual_seed(config.seed)

    if config.log_path:
        open(config.log_path, "w", encoding="utf-8").close()

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_ds, val_ds, test_ds, class_weights = load_splits(csv_path=config.csv_path)

    train_loader, val_loader, test_loader = make_dataloaders(
        train_ds,
        val_ds,
        test_ds,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_length=config.max_length
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=3,
        output_attentions=True,
    )
    model.to(device)

    class_weights = class_weights.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val_f1 = 0.0
    best_state = None

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", leave=False)
        for idx, batch in enumerate(progress):
            batch = _to_device(batch, device)
            labels = batch.pop("labels")

            outputs = model(**batch)
            logits = outputs.logits
            loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            progress.set_postfix(loss=loss.item())
            print(f"We are now at step {idx} of epoch {epoch}")

        train_loss_avg = epoch_loss / max(len(train_loader), 1)

        val_acc, val_macro_f1, val_per_class = _evaluate(model, val_loader, device)
        _log_metrics(
            config.log_path,
            {
                "epoch": epoch + 1,
                "split": "val",
                "train_loss": train_loss_avg,
                "val_accuracy": val_acc,
                "val_macro_f1": val_macro_f1,
                "val_per_class_f1": val_per_class,
            },
        )
        print(f"We have now completed epoch {epoch}")
        print("=" * 80, "\n")
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, test_macro_f1, test_per_class = _evaluate(model, test_loader, device)

    _log_metrics(
        config.log_path,
        {
            "epoch": config.epochs,
            "split": "test",
            "test_accuracy": test_acc,
            "test_macro_f1": test_macro_f1,
            "test_per_class_f1": test_per_class,
            "best_val_macro_f1": best_val_f1,
        },
    )

    metrics = {
        "val_macro_f1": best_val_f1,
        "test_accuracy": test_acc,
        "test_macro_f1": test_macro_f1,
    }
    metrics.update({f"test_f1_class_{k}": v for k, v in test_per_class.items()})
    return metrics

def build_collate_and_loader(
    csv_path: str,
    model_name: str = "roberta-base",
    batch_size: int = 8,
    max_length: int = 512,
    num_workers: int = 2,
) -> Tuple[ConversationDataset, ConversationDataset, ConversationDataset, DataLoader, DataLoader, DataLoader]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds, val_ds, test_ds, _ = load_splits(csv_path=csv_path)
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
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
