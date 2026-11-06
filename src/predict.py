"""Load the saved pipeline and make predictions for new house records."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.data import FEATURE_COLUMNS
from src.train import ROOT_DIR

DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "house_price_pipeline.joblib"


@lru_cache(maxsize=4)
def _load_model_cached(model_path: str):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}. Train the model first.")
    return joblib.load(path)


def load_model(model_path: str | Path | None = None):
    selected_path = Path(model_path or os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))).resolve()
    return _load_model_cached(str(selected_path))


def predict_records(records, model_path: str | Path | None = None) -> list[float]:
    frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    missing = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction input is missing features: {missing}")
    frame = frame[FEATURE_COLUMNS]
    predictions = load_model(model_path).predict(frame)
    return [float(value) for value in predictions]


if __name__ == "__main__":
    example = {
        "area_sqft": 1450,
        "bedrooms": 3,
        "bathrooms": 2,
        "age_years": 8,
        "distance_to_city_km": 6.5,
        "parking_spaces": 1,
    }
    print({"prediction_lakh": predict_records([example])[0]})
