import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from classifier.train import TrainConfig, train_model
from classifier.data import DataConfig
from classifier.attention import AttentionConfig, run_attention_dump
from diffuser.preprocessing.label_speakers import label_speakers
from diffuser.preprocessing.fuzzy_embedding import create_embeddings
from diffuser.vq_vae.train import VQVAEConfig, train_vqvae
from diffuser.vq_vae.export_codes import ExportCodesConfig, export_codes


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def _build_data_cfg(cfg: Dict[str, Any]) -> DataConfig:
    return DataConfig(**cfg)


def _build_train_cfg(cfg: Dict[str, Any], data_cfg: DataConfig, run_dir: Path) -> TrainConfig:
    classifier_dir = run_dir / "classifier"
    model_dir = classifier_dir / cfg.get("model_subdir", "model")
    log_path = classifier_dir / cfg.get("log_filename", "metrics.jsonl")

    return TrainConfig(
        data=data_cfg,
        model_name=cfg.get("model_name", "roberta-base"),
        batch_size=int(cfg.get("batch_size", 8)),
        max_length=int(cfg.get("max_length", 512)),
        epochs=int(cfg.get("epochs", 3)),
        learning_rate=float(cfg.get("learning_rate", 2e-5)),
        seed=int(cfg.get("seed", 42)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.1)),
        log_path=str(log_path),
        output_dir=str(model_dir),
    )


def _build_attention_cfg(cfg: Dict[str, Any], data_cfg: DataConfig, run_dir: Path, model_dir: Path) -> AttentionConfig:
    attention_dir = run_dir / "attention"
    output_path = attention_dir / cfg.get("output_filename", "attention_scores.txt")
    return AttentionConfig(
        data=data_cfg,
        model_dir=str(cfg.get("model_dir", model_dir)),
        output_path=str(output_path),
        sample_size=int(cfg.get("sample_size", 20)),
        max_length=int(cfg.get("max_length", 512)),
        layer=int(cfg.get("layer", -1)),
        seed=int(cfg.get("seed", 42)),
    )


def _build_vqvae_cfg(cfg: Dict[str, Any], run_dir: Path) -> VQVAEConfig:
    vq_dir = run_dir / "vqvae"
    ckpt_name = cfg.get("checkpoint_name", "vqvae.pt")
    ckpt_path = vq_dir / cfg.get("checkpoint_subdir", "checkpoints") / ckpt_name
    npy_path = vq_dir / cfg.get("npy_path", "embeddings.npy")

    return VQVAEConfig(
        jsonl_path=str(cfg.get("jsonl_path", "sentence_embeddings.jsonl")),
        npy_path=str(npy_path),
        checkpoint_path=str(ckpt_path),
        batch_size=int(cfg.get("batch_size", 256)),
        epochs=int(cfg.get("epochs", 10)),
        learning_rate=float(cfg.get("learning_rate", 1e-3)),
        log_every=int(cfg.get("log_every", 100)),
        num_codes=int(cfg.get("num_codes", 128)),
        latent_dim=int(cfg.get("latent_dim", 128)),
        hidden=int(cfg.get("hidden", 256)),
        beta=float(cfg.get("beta", 0.25)),
        l2_norm_embs=bool(cfg.get("l2_norm_embs", True)),
        seed=int(cfg.get("seed", 42)),
    )


def _build_export_cfg(cfg: Dict[str, Any], run_dir: Path, ckpt_path: Path, npy_path: Path) -> ExportCodesConfig:
    export_dir = run_dir / "vqvae"
    jsonl_out = export_dir / cfg.get("jsonl_out", "sentence_cluster_ids.jsonl")
    return ExportCodesConfig(
        jsonl_in=str(cfg.get("jsonl_in", "sentence_embeddings.jsonl")),
        npy_path=str(cfg.get("npy_path", npy_path)),
        checkpoint_path=str(cfg.get("checkpoint_path", ckpt_path)),
        jsonl_out=str(jsonl_out),
        batch_size=int(cfg.get("batch_size", 512)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run conv-transformer tasks from YAML config")
    parser.add_argument(
        "--task",
        choices=["classifier_train", "attention", "vqvae_train", "export_codes"],
        default="vqvae_train",
        help="Task to run (defaults to vqvae_train)",
    )
    parser.add_argument("--run-version", dest="run_version", help="Override run_version label (e.g., v0.2)")
    args = parser.parse_args()

    cfg_path = Path("config.yaml")
    cfg = _load_yaml(cfg_path)

    core_cfg = cfg.get("core", {})
    run_version = args.run_version or core_cfg.get("run_version", "v0.1")
    output_root = Path(core_cfg.get("output_root", "runs"))
    run_dir = output_root / run_version
    run_dir.mkdir(parents=True, exist_ok=True)

    # Task now comes from CLI (with a sane default) rather than config to avoid missing keys.
    task = args.task

    data_cfg = _build_data_cfg(cfg.get("data", {}))

    classifier_cfg_raw = cfg.get("classifier", {})
    train_cfg = _build_train_cfg(classifier_cfg_raw, data_cfg, run_dir)

    attention_cfg_raw = cfg.get("attention", {})
    classifier_model_dir = Path(train_cfg.output_dir)
    attention_cfg = _build_attention_cfg(attention_cfg_raw, data_cfg, run_dir, classifier_model_dir)

    vqvae_cfg_raw = cfg.get("vqvae", {})
    vqvae_cfg = _build_vqvae_cfg(vqvae_cfg_raw, run_dir)

    export_cfg_raw = cfg.get("export_codes", {})
    export_cfg = _build_export_cfg(export_cfg_raw, run_dir, Path(vqvae_cfg.checkpoint_path), Path(vqvae_cfg.npy_path))

    if task == "classifier_train":
        metrics = train_model(train_cfg)
        print(json.dumps(metrics, indent=2))
    elif task == "attention":
        run_attention_dump(attention_cfg)
    elif task == "vqvae_train":
        train_vqvae(vqvae_cfg)
    elif task == "export_codes":
        export_codes(export_cfg)
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    main()
