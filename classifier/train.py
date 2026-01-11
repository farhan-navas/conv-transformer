import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from data import DataConfig, load_code_splits, make_code_dataloaders
from code_model import CodeClassifier, CodeModelConfig

@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    codes_jsonl: str = "sentence_cluster_ids.jsonl"
    vocab_size: int = 128
    pad_id: int = 128
    cls_id: int = 129
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 32
    max_length: int = 256
    epochs: int = 5
    learning_rate: float = 1e-4
    seed: int = 42
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    log_path: str = "metrics.jsonl"
    output_dir: str = "model_out"

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
            logits = model(batch["input_ids"], batch["attention_mask"])
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
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if config.log_path:
        Path(config.log_path).parent.mkdir(parents=True, exist_ok=True)
        open(config.log_path, "w", encoding="utf-8").close()

    train_ds, val_ds, test_ds, class_weights = load_code_splits(
        codes_jsonl=config.codes_jsonl,
        data_cfg=config.data,
    )

    train_loader, val_loader, test_loader = make_code_dataloaders(
        train_ds,
        val_ds,
        test_ds,
        batch_size=config.batch_size,
        max_length=config.max_length,
        pad_id=config.pad_id,
        cls_id=config.cls_id,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = CodeModelConfig(
        vocab_size=config.vocab_size,
        pad_id=config.pad_id,
        cls_id=config.cls_id,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
        max_length=config.max_length,
        num_labels=len(config.data.label_map),
    )
    model = CodeClassifier(model_cfg)
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

            logits = model(batch["input_ids"], batch["attention_mask"])
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

    torch.save(
        {
            "model_state": model.state_dict(),
            "model_cfg": model_cfg.__dict__,
        },
        out_dir / "model.pt",
    )

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
