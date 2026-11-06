# MLOps Lab 7-9: House-Price Prediction Service

This project converts a small machine-learning workflow into a reproducible application. It uses an **artificial, locally generated dataset**; no real property, customer or personal data is included.

## 1. Project structure

```text
mlops-lab-7-9/
├── data/raw/housing.csv
├── models/house_price_pipeline.joblib
├── reports/metrics.json
├── reports/evaluation_report.json
├── configs/config.yaml
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── generate_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── api.py
├── tests/
│   ├── test_pipeline.py
│   └── test_api.py
├── Dockerfile
├── Dockerfile.mlflow
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── dvc.yaml
├── requirements.txt
└── requirements-mlflow.txt
```

## 2. Local setup (Windows PowerShell)

Python 3.11 is the target runtime.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Generate the artificial dataset and train the pipeline:

```powershell
python -m src.generate_data
python -m src.train
python -m src.evaluate
```

The training command saves the **whole inference pipeline** (feature engineering, imputation, scaling and the Random Forest estimator) to `models/house_price_pipeline.joblib`. It also writes metrics to `reports/metrics.json` and `reports/evaluation_report.json`.

Run automated tests:

```powershell
python -m pytest -q
```

The tests expect the model to have been generated first. The expected result for this checked-in dataset and configuration is **7 passed**. Exact floating-point values can vary slightly with library versions.

## 3. Run the REST API

```powershell
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

Open the automatically generated Swagger UI at `http://127.0.0.1:8000/docs`.

Health request:

```powershell
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/health
```

Prediction request:

```powershell
$body = @{
  area_sqft = 1450
  bedrooms = 3
  bathrooms = 2
  age_years = 8
  distance_to_city_km = 6.5
  parking_spaces = 1
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" -Body $body
```

The sample response includes `prediction_lakh`, `currency` and `unit`. The numeric prediction is model-dependent; the report records the value measured during its local test run.

## 4. Docker build and run

First train the model so the artifact exists. Install and start Docker Desktop, then run these commands from the project root:

```powershell
docker build -t ml-inference-api:1.0 .
docker image ls
docker run --name ml-api -p 8000:8000 ml-inference-api:1.0
```

In another terminal, test `http://localhost:8000/health` and `http://localhost:8000/docs`. To stop and remove the container:

```powershell
docker ps
docker logs ml-api
docker stop ml-api
docker rm ml-api
```

If host port 8000 is already in use, use `-p 8001:8000` and open port 8001 in the browser.

## 5. Docker Compose with MLflow (optional extension)

The Compose design builds two services: FastAPI and an MLflow Tracking Server. It stores the MLflow SQLite backend database and model artifacts in named Docker volumes. The MLflow server version is pinned to `3.15.0` in `Dockerfile.mlflow`; update and test this version deliberately rather than using an unpinned `latest` tag.

```powershell
docker compose build
docker compose up -d mlflow
```

To log a local training run to the Compose MLflow service, install the optional client and set the host-side tracking URI:

```powershell
python -m pip install -r requirements-mlflow.txt
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
python -m src.train
Remove-Item Env:MLFLOW_TRACKING_URI
```

Build and run the API service after training:

```powershell
docker compose up --build -d api
docker compose ps
docker compose logs
```

The API container uses `http://mlflow:5000` as the tracking-server address on the Compose network. To stop services while keeping named volumes:

```powershell
docker compose down
```

To remove the services **and** their stored MLflow database/artifacts:

```powershell
docker compose down -v
```

For a real team deployment, use a managed database/object store, authentication, restricted network access and secrets management. The sample Compose setup is for a local lab only.

## 6. Dataset versioning with DVC

The included `dvc.yaml` defines a reproducible training stage. To begin tracking the raw dataset with DVC:

```powershell
python -m pip install dvc
dvc init
dvc add data/raw/housing.csv
dvc repro
dvc status
```

Commit the `.dvc` metadata and the DVC-generated `.gitignore` file, not a second independent copy of the dataset in Git. Configure a remote with `dvc remote add -d storage <remote-url>` when a team-approved remote is available. Do not add passwords or access tokens to the repository.

## 7. Input schema

| Field | Type | Rule / unit |
|---|---|---|
| `area_sqft` | number | Greater than 0; square feet |
| `bedrooms` | integer | 1 to 20 |
| `bathrooms` | number | Greater than 0; maximum 20 |
| `age_years` | number | 0 to 150 years |
| `distance_to_city_km` | number | 0 to 300 km |
| `parking_spaces` | integer | 0 to 10 |

The model predicts a synthetic estimated price in lakh Indian rupees. It is an educational example, not a real-estate valuation tool.

## 8. Reproducibility and safety notes

- The random seed and test size are stored in `configs/config.yaml`.
- The hold-out test split is not used to fit the imputer, scaler or estimator.
- The saved Scikit-learn Pipeline applies the same transformations at prediction time.
- `MODEL_PATH` can be set as an environment variable to select another model artifact.
- `.dockerignore` excludes raw data, local virtual environments, test files and local reports from the API image.
- Never add `.env` files, passwords, API tokens or private keys to Git or Docker images.
- Joblib model files should only be loaded from trusted sources because they use Python object serialization.

## 9. Practical evidence and report

The `evidence/screenshots/` folder contains six Windows 11-style evidence views:

- Model training and held-out evaluation metrics.
- FastAPI health and prediction responses.
- Swagger UI endpoint view.
- Invalid input validation returning HTTP 422.
- Automated test output (`7 passed`).
- Docker availability status (pending, not a successful Docker run).

The completed report is included as `Report.docx` and `Report.pdf`. The original brief is stored at `docs/Original_MLOps_Lab_Assignment_Brief.docx`.

**Screenshot integrity note:** The Windows-style frames are recreated from recorded local execution outputs and API responses. They are not direct captures from a physical Windows laptop. Docker CLI/Engine was unavailable in the authoring workspace, so no successful Docker build or container run is claimed. Follow `evidence/DOCKER_SCREENSHOT_GUIDE.txt` on a computer with Docker installed, then replace the pending-status view with genuine screenshots from that computer.

### Python runtime note

The Dockerfile follows the assignment example and uses Python 3.11. The prepared project was executed and tested in this workspace with Python 3.13.5. Before building the Docker image on your computer, create the project environment with Python 3.11 (as shown in the setup commands), install `requirements.txt`, and rerun `python -m src.train`. This regenerates the saved Joblib pipeline in the same Python runtime that the container will use. Then build the Docker image.
