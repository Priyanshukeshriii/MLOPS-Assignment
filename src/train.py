"""Train, evaluate and save the complete inference pipeline."""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.data import add_engineered_features, clean_data, load_data, split_data

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT_DIR / "configs" / "config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT_DIR / path


def build_pipeline(model_config: dict, random_state: int) -> Pipeline:
    numeric_preprocessing = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_preprocessing, make_column_selector(dtype_include="number"))
        ],
        remainder="drop",
    )
    regressor = RandomForestRegressor(
        n_estimators=int(model_config.get("n_estimators", 100)),
        max_depth=model_config.get("max_depth"),
        min_samples_split=int(model_config.get("min_samples_split", 2)),
        min_samples_leaf=int(model_config.get("min_samples_leaf", 1)),
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("feature_engineering", FunctionTransformer(add_engineered_features, validate=False)),
            ("preprocessing", preprocessor),
            ("regressor", regressor),
        ]
    )


def calculate_metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def optional_mlflow_logging(config: dict, metrics: dict, model: Pipeline, model_path: Path) -> None:
    """Log a run only when a tracking URI is explicitly configured and MLflow exists."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("MLflow tracking skipped: MLFLOW_TRACKING_URI is not set.")
        return
    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("MLOps-House-Price-Pipeline")
        with mlflow.start_run():
            mlflow.log_params(config.get("model", {}))
            mlflow.log_param("random_state", config["training"]["random_state"])
            mlflow.log_param("test_size", config["training"]["test_size"])
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(model_path), artifact_path="model-file")
            mlflow.sklearn.log_model(model, artifact_path="sklearn-pipeline")
        print(f"MLflow run logged to {tracking_uri}.")
    except ImportError:
        print("MLflow tracking requested, but MLflow is not installed. Install requirements-mlflow.txt.")
    except Exception as exc:  # Keep a tracking outage from hiding a successful local training run.
        print(f"Warning: MLflow logging was not completed: {type(exc).__name__}: {exc}")


def train_model(config_path: str | Path = DEFAULT_CONFIG) -> dict:
    config = load_config(config_path)
    data_path = resolve_path(config["paths"]["data"])
    model_path = resolve_path(config["paths"]["model"])
    metrics_path = resolve_path(config["paths"]["metrics"])
    report_path = resolve_path(config["paths"]["evaluation_report"])

    frame = clean_data(load_data(data_path))
    training_cfg = config["training"]
    X_train, X_test, y_train, y_test = split_data(
        frame,
        target_column=config["data"]["target_column"],
        test_size=float(training_cfg["test_size"]),
        random_state=int(training_cfg["random_state"]),
    )

    pipeline = build_pipeline(config["model"], int(training_cfg["random_state"]))
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metrics = calculate_metrics(y_test, predictions)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    metrics_payload = {
        **metrics,
        "rows_total": int(len(frame)),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "features": list(X_train.columns),
        "target": config["data"]["target_column"],
        "model_type": config["model"]["type"],
        "random_state": int(training_cfg["random_state"]),
        "test_size": float(training_cfg["test_size"]),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    report_path.write_text(
        json.dumps(
            {
                "project": config["project"]["name"],
                "evaluation_method": "held-out test set; preprocessing fitted on training data only",
                "metrics": metrics_payload,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    optional_mlflow_logging(config, metrics, pipeline, model_path)
    print(f"Training completed. Model saved to: {model_path}")
    print(json.dumps(metrics_payload, indent=2))
    return metrics_payload


def main() -> None:
    train_model()


if __name__ == "__main__":
    main()
