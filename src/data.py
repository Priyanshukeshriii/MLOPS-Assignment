"""Data loading, validation, cleaning, feature engineering and splitting."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "age_years",
    "distance_to_city_km",
    "parking_spaces",
]
TARGET_COLUMN = "price_lakh"
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def generate_synthetic_data(
    output_path: str | Path,
    n_rows: int = 1200,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a deterministic, synthetic dataset for demonstration and testing.

    The records are artificial and do not describe real houses or people.
    Prices are expressed in Indian lakh rupees.
    """
    if n_rows < 50:
        raise ValueError("n_rows must be at least 50 for a useful train/test split.")

    rng = np.random.default_rng(random_state)
    area = rng.integers(450, 3001, size=n_rows).astype(float)
    bedrooms = np.clip(np.rint(area / 520 + rng.normal(0, 0.8, n_rows)), 1, 6)
    bathrooms = np.clip(np.rint(bedrooms * 0.72 + rng.normal(0.4, 0.55, n_rows)), 1, 5)
    age = rng.integers(0, 41, size=n_rows).astype(float)
    distance = np.round(rng.uniform(0.5, 35.0, size=n_rows), 2)
    parking = np.clip(np.rint((bedrooms - 1) / 2 + rng.normal(0.4, 0.7, n_rows)), 0, 4)

    noise = rng.normal(0, 5.0, size=n_rows)
    price = (
        9.0
        + 0.034 * area
        + 1.45 * bedrooms
        + 1.05 * bathrooms
        - 0.22 * age
        - 0.48 * distance
        + 0.95 * parking
        + noise
    )
    price = np.maximum(price, 8.0)

    frame = pd.DataFrame(
        {
            "area_sqft": area,
            "bedrooms": bedrooms.astype(int),
            "bathrooms": bathrooms.astype(int),
            "age_years": age,
            "distance_to_city_km": distance,
            "parking_spaces": parking.astype(int),
            TARGET_COLUMN: np.round(price, 2),
        }
    )

    # Add a small number of missing feature values to exercise the imputer.
    missing_rng = np.random.default_rng(random_state + 1)
    for column in ["bathrooms", "distance_to_city_km"]:
        indices = missing_rng.choice(frame.index, size=max(1, n_rows // 100), replace=False)
        frame.loc[indices, column] = np.nan

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def load_data(path: str | Path) -> pd.DataFrame:
    """Read a CSV file and check that all required columns are present."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}. Generate it with generate_synthetic_data()."
        )
    frame = pd.read_csv(file_path)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    return frame[REQUIRED_COLUMNS].copy()


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean rows and types while leaving feature imputation to the ML pipeline."""
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Input data is missing required columns: {missing_columns}")

    cleaned = frame[REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.drop_duplicates()
    # A target value is needed for supervised learning; missing feature values are
    # retained and handled by SimpleImputer fitted on the training split only.
    cleaned = cleaned.dropna(subset=[TARGET_COLUMN])
    cleaned = cleaned[
        (cleaned["area_sqft"].isna() | (cleaned["area_sqft"] > 0))
        & (cleaned["bedrooms"].isna() | (cleaned["bedrooms"] >= 1))
        & (cleaned["bathrooms"].isna() | (cleaned["bathrooms"] >= 1))
        & (cleaned["age_years"].isna() | (cleaned["age_years"] >= 0))
        & (cleaned["distance_to_city_km"].isna() | (cleaned["distance_to_city_km"] >= 0))
        & (cleaned["parking_spaces"].isna() | (cleaned["parking_spaces"] >= 0))
        & (cleaned[TARGET_COLUMN] > 0)
    ]
    if cleaned.empty:
        raise ValueError("No valid rows remain after data cleaning.")
    return cleaned.reset_index(drop=True)


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add derived features using the same function during training and inference."""
    result = frame.copy()
    bedrooms_safe = pd.to_numeric(result["bedrooms"], errors="coerce").clip(lower=1)
    bathrooms_safe = pd.to_numeric(result["bathrooms"], errors="coerce").clip(lower=1)
    result["area_per_bedroom"] = result["area_sqft"] / bedrooms_safe
    result["bathroom_bedroom_ratio"] = bathrooms_safe / bedrooms_safe
    return result


def split_data(
    frame: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split records before fitting preprocessing to prevent data leakage."""
    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' is not present.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    X = frame.drop(columns=[target_column])
    y = frame[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
