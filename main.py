from __future__ import annotations

import argparse
import json

from conv_transformer.train import TrainConfig, train_model


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train a RoBERTa dialogue outcome classifier")
	parser.add_argument("csv_path", help="Path to CSV with columns: outcome, transcript")
	parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
	parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
	parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
	parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
	parser.add_argument("--model-name", default="roberta-base", help="HF model name or path")
	parser.add_argument("--num-workers", type=int, default=2, help="Dataloader workers")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	config = TrainConfig(
		csv_path=args.csv_path,
		model_name=args.model_name,
		batch_size=args.batch_size,
		max_length=args.max_length,
		epochs=args.epochs,
		learning_rate=args.learning_rate,
		num_workers=args.num_workers,
	)
	metrics = train_model(config)
	print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
	main()
