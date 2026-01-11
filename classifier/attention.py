from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import torch
from torch.nn.functional import softmax

from .code_model import CodeClassifier, CodeModelConfig
from .data import DataConfig, _load_code_examples


def _sample_indices(n: int, sample_size: int, seed: int) -> List[int]:
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng)
    return perm[: min(sample_size, n)].tolist()


def _normalize(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = scores * mask
    total = masked.sum() + 1e-12
    return masked / total


@dataclass
class AttentionConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model_dir: str = "model_out"
    codes_jsonl: str = "sentence_cluster_ids.jsonl"
    output_path: str = "attention_scores.txt"
    sample_size: int = 20
    max_length: int = 512
    layer: int = -1
    seed: int = 42


def run_attention_dump(config: AttentionConfig) -> None:
    out_path = Path(config.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(Path(config.model_dir) / "model.pt", map_location="cpu")
    model_cfg = CodeModelConfig(**ckpt["model_cfg"])
    model = CodeClassifier(model_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    examples = _load_code_examples(config.codes_jsonl, config.data)
    id2label: Dict[int, str] = {v: k for k, v in config.data.label_map.items()}

    with torch.no_grad(), open(out_path, "w", encoding="utf-8") as f:
        for idx in _sample_indices(len(examples), config.sample_size, config.seed):
            ex = examples[idx]
            codes = list(ex.codes)
            if model_cfg.cls_id is not None:
                codes = [model_cfg.cls_id] + codes
            if len(codes) > model_cfg.max_length:
                codes = codes[-model_cfg.max_length :]

            input_ids = torch.tensor(codes, dtype=torch.long).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)

            logits, attentions = model(input_ids, attention_mask, return_attentions=True)  # type: ignore
            probs = softmax(logits, dim=-1)[0].tolist()
            pred_id = int(torch.argmax(logits, dim=-1).item())
            pred_label = id2label.get(pred_id, str(pred_id))

            attn_layer = attentions[config.layer]
            cls_attn = attn_layer.mean(dim=1)[0, 0]  # average heads, take CLS query

            mask = attention_mask[0].float()
            if model_cfg.cls_id is not None:
                mask[0] = 0.0  # drop CLS from distribution
            scores = _normalize(cls_attn, mask)

            tokens = [str(t) for t in codes]
            token_scores = list(zip(tokens, scores.tolist()))

            f.write(
                f"Example {idx}: gold={id2label.get(ex.label, ex.label)} pred={pred_label} probs={probs}\n"
                f"Tokens+attn: {token_scores}\n"
                f"{'=' * 60}\n\n"
            )