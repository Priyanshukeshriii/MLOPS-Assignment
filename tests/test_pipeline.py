from pathlib import Path

import pandas as pd

from src.data import FEATURE_COLUMNS, clean_data, load_data
from src.predict import predict_records
from src.train import ROOT_DIR


def test_dataset_has_expected_columns_and_valid_rows():
    data_path = ROOT_DIR / "data" / "raw" / "housing.csv"
    frame = clean_data(load_data(data_path))
    assert set(FEATURE_COLUMNS + ["price_lakh"]).issubset(frame.columns)
    assert len(frame) >= 1000
    assert (frame["price_lakh"] > 0).all()


def test_prediction_returns_one_value_per_record():
    sample = {
        "area_sqft": 1450,
        "bedrooms": 3,
        "bathrooms": 2,
        "age_years": 8,
        "distance_to_city_km": 6.5,
        "parking_spaces": 1,
    }
    outputs = predict_records([sample, sample])
    assert len(outputs) == 2
    assert all(isinstance(value, float) for value in outputs)
    assert all(value > 0 for value in outputs)
