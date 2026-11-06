# Practical evidence folder

The report embeds the following Windows 11-style evidence views at the relevant sections. The image content is recreated from the recorded local execution outputs and API responses; these are not direct screen captures from a physical Windows laptop.

## Local execution evidence

- `screenshots/01_windows_model_training.png` - model training and held-out evaluation metrics.
- `screenshots/02_windows_api_health_prediction.png` - local GET `/health` and POST `/predict` smoke-test results.
- `screenshots/03_windows_swagger_ui.png` - Swagger UI-style view of the API endpoints and recorded responses.
- `screenshots/04_windows_invalid_input_422.png` - invalid negative area value rejected with HTTP 422.
- `screenshots/05_windows_automated_tests.png` - automated test output showing 7 passed.

## Docker status - pending

- `screenshots/06_windows_docker_environment_check.png` - status-only view showing that the Docker CLI was unavailable in the authoring environment. It is **not** evidence of a successful Docker image build or container run.
- `DOCKER_SCREENSHOT_GUIDE.txt` contains the commands to run on a computer with Docker Desktop/Engine installed. Capture genuine build, container, health, prediction and log screenshots there before claiming Docker execution is complete.

The training dataset is generated synthetic data. The house-price output is educational and is not a real property valuation.
