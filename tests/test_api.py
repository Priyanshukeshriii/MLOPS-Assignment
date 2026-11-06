from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

VALID_PAYLOAD = {
    "area_sqft": 1450,
    "bedrooms": 3,
    "bathrooms": 2,
    "age_years": 8,
    "distance_to_city_km": 6.5,
    "parking_spaces": 1,
}


def test_health_endpoint_reports_model_file():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_valid_prediction_request():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction_lakh"] > 0
    assert body["unit"] == "lakh"


def test_missing_required_field_is_rejected():
    payload = VALID_PAYLOAD.copy()
    payload.pop("area_sqft")
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_wrong_data_type_is_rejected():
    payload = VALID_PAYLOAD.copy()
    payload["bedrooms"] = "three"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_invalid_business_value_is_rejected():
    payload = VALID_PAYLOAD.copy()
    payload["area_sqft"] = -10
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
