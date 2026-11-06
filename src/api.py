"""FastAPI application exposing the saved house-price inference pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.predict import DEFAULT_MODEL_PATH, predict_records

app = FastAPI(
    title="House Price Prediction API",
    version="1.0.0",
    description="A demonstration API for a synthetic house-price regression model.",
)


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    area_sqft: Annotated[float, Field(gt=0, le=20000, description="Built-up area in square feet")]
    bedrooms: Annotated[int, Field(ge=1, le=20)]
    bathrooms: Annotated[float, Field(gt=0, le=20)]
    age_years: Annotated[float, Field(ge=0, le=150)]
    distance_to_city_km: Annotated[float, Field(ge=0, le=300)]
    parking_spaces: Annotated[int, Field(ge=0, le=10)]


@app.get("/health", tags=["service"])
def health():
    """Return service status and whether the model artifact is present."""
    model_exists = Path(DEFAULT_MODEL_PATH).exists()
    return {
        "status": "ok" if model_exists else "model_not_found",
        "model_loaded": model_exists,
    }


@app.post("/predict", tags=["inference"])
def predict(payload: PredictionInput):
    """Predict an estimated house price in lakh Indian rupees."""
    try:
        prediction = predict_records([payload.model_dump()])[0]
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Prediction model is not available. Train or mount the model first.")
    except ValueError:
        raise HTTPException(status_code=422, detail="The supplied values do not satisfy the model input requirements.")
    except Exception:
        # Do not return local file paths, stack traces or internal exception details.
        raise HTTPException(status_code=500, detail="Prediction could not be completed.")

    return {
        "prediction_lakh": round(prediction, 2),
        "currency": "INR",
        "unit": "lakh",
        "model": "house_price_pipeline",
    }
