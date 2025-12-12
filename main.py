import json

from classifier.train import TrainConfig, train_model

def main() -> None:
	config = TrainConfig(
		csv_path="data.csv"
	)
	metrics = train_model(config)
	print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
	main()
