"""Evaluate the saved inference pipeline on a held-out test split."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import yaml

from src.data import clean_data, load_data, split_data
from src.train import DEFAULT_CONFIG, calculate_metrics, resolve_path


def evaluate_model(config_path: str | Path = DEFAULT_CONFIG) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    frame = clean_data(load_data(resolve_path(config["paths"]["data"])))
    _, X_test, _, y_test = split_data(
        frame,
        target_column=config["data"]["target_column"],
        test_size=float(config["training"]["test_size"]),
        random_state=int(config["training"]["random_state"]),
    )
    model_path = resolve_path(config["paths"]["model"])
    if not model_path.exists():
        raise FileNotFoundError("Saved model not found. Run `python -m src.train` first.")

    pipeline = joblib.load(model_path)
    predictions = pipeline.predict(X_test)
    metrics = calculate_metrics(y_test, predictions)
    report_path = resolve_path(config["paths"]["evaluation_report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({"evaluation": "saved model on reproducible held-out split", "metrics": metrics}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    evaluate_model()
