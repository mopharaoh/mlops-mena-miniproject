# MLOps Mini Project 1 — House Prices Prediction API

A production-oriented machine learning project that transforms a House Prices regression notebook into a structured, testable, package-based FastAPI service and Docker container.

The project follows the MLOps workflow covered in Module 1:

- Refactoring notebook code into reusable Python modules
- Model training and persistence
- Structured JSON logging
- Pickle and ONNX serialization
- Prediction API with FastAPI and Pydantic
- Automated testing with pytest
- Docker multi-stage build
- Docker Compose deployment
- Non-root container execution
- Container health checks

---

## 1. Project Overview

The original project started as a House Prices regression notebook.

The goal of this mini project was to move from a notebook-oriented workflow toward a production-style machine learning application.

The application:

1. Loads the House Prices dataset.
2. Separates features and target.
3. Splits the data into training and validation sets.
4. Handles numerical and categorical features.
5. Trains a Linear Regression model.
6. Saves the trained model as a Pickle artifact.
7. Exports the model to ONNX.
8. Provides a FastAPI prediction service.
9. Validates API requests with Pydantic.
10. Adds structured JSON logging and request correlation IDs.
11. Tests the application with pytest.
12. Packages the application into a multi-stage Docker image.
13. Runs the API as a non-root user.
14. Provides a Docker health check.

---

## 2. Dataset

The project uses the Kaggle/Ames House Prices dataset.

The target variable is:

```text
SalePrice
```

The dataset contains 1460 training rows.

The `Id` column is removed before model training because it is an identifier rather than a predictive feature.

---

## 3. Machine Learning Pipeline

The model uses:

```text
House Prices Dataset
        |
        v
Feature / Target Split
        |
        v
Train / Validation Split
        |
        +----------------------+
        |                      |
        v                      v
Numerical Features       Categorical Features
        |                      |
        v                      v
Median Imputation       Training-set Mode
        |                      |
        v                      v
StandardScaler          Missing-value Filling
        |                      |
        +----------+-----------+
                   |
                   v
             OneHotEncoder
                   |
                   v
           Linear Regression
                   |
                   v
             Prediction
```

### Numerical features

Numerical missing values are handled using:

```text
SimpleImputer(strategy="median")
```

and then scaled using:

```text
StandardScaler
```

### Categorical features

Categorical missing values are filled using values calculated from the training data.

Categorical encoding uses:

```text
OneHotEncoder(handle_unknown="ignore")
```

This allows unseen categorical values during inference to be ignored instead of causing the prediction request to fail.

---

## 4. Project Structure

```text
mlops-menaProjects/
│
├── data/
│   └── raw/
│       └── train.csv
│
├── models/
│   ├── model.pkl
│   └── model.onnx
│
├── src/
│   └── prodml/
│       ├── __init__.py
│       │
│       ├── api/
│       │   ├── __init__.py
│       │   ├── main.py
│       │   └── schemas.py
│       │
│       ├── benchmark.py
│       ├── config.py
│       ├── data.py
│       ├── export.py
│       ├── features.py
│       ├── logging_conf.py
│       ├── predict.py
│       └── train.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_features.py
│   ├── test_logging.py
│   ├── test_predict.py
│   └── test_serialization.py
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── pyproject.toml
├── uv.lock
└── README.md
```

---

# 5. Main Components

## `data.py`

Responsible for dataset-related operations:

- Loading the CSV dataset
- Separating features and target
- Removing the `Id` column
- Calculating categorical fill values
- Filling categorical missing values
- Splitting data into training and validation sets

---

## `features.py`

Responsible for preprocessing.

It contains:

- Categorical missing-value handling
- Categorical fill-value calculation
- Numerical imputation
- Numerical scaling
- One-hot encoding
- Construction of the sklearn `ColumnTransformer`

---

## `train.py`

Responsible for model training.

The training process:

1. Loads the dataset.
2. Separates features and target.
3. Creates training and validation sets.
4. Detects numerical and categorical features.
5. Calculates categorical fill values using training data.
6. Builds the preprocessing pipeline.
7. Trains `LinearRegression`.
8. Calculates MAE and RMSE.
9. Saves the trained model to:

```text
models/model.pkl
```

---

## `predict.py`

Contains the `HousePricePredictor` inference interface.

The predictor owns:

- The trained sklearn pipeline
- Categorical fill values
- Model version

It provides:

```text
predict_one()
predict_batch()
prepare_for_onnx()
```

It also validates incoming feature names and raises:

```text
FeatureValidationError
```

when unknown feature names are received.

The prediction methods are timed using the `timed` decorator.

---

# 6. Model Serialization

Two model formats are used.

## Pickle

The trained sklearn predictor is persisted using:

```text
joblib
```

and saved as:

```text
models/model.pkl
```

The Pickle artifact is used for the API inference service.

### Security note

Pickle files must only be loaded from trusted sources.

Loading an untrusted Pickle file can execute arbitrary code.

Therefore, untrusted `.pkl` files should never be loaded.

---

## ONNX

The trained sklearn model is also exported to:

```text
models/model.onnx
```

ONNX Runtime is used for inference benchmarking.

The project verifies prediction parity between the Pickle/sklearn model and ONNX Runtime using:

```text
numpy.allclose(..., atol=1e-4)
```

---

# 7. Serialization Benchmark

The same validation dataset is used for both serialization formats.

Benchmark configuration:

```text
Dataset rows: 292
Warmup runs: 20
Measured repetitions: 100
```

Measured results:

| Format | Mean latency | P95 latency |
|---|---:|---:|
| Pickle / sklearn | 20.129 ms | 21.019 ms |
| ONNX Runtime | 1.428 ms | 1.708 ms |

The benchmark demonstrates the inference latency difference between the two runtime approaches on the same validation data and hardware environment.

---

# 8. Structured Logging

The project uses JSON structured logging.

Log records contain fields such as:

```text
asctime
levelname
name
message
correlation_id
```

A request receives a UUID4 correlation ID through FastAPI middleware.

The correlation ID is also returned to the client through:

```text
X-Request-ID
```

This allows a request to be traced between the API response and application logs.

The project avoids using `print()` inside the application source.

---

# 9. FastAPI

The API is implemented using FastAPI.

The model is loaded once during application startup using FastAPI's lifespan mechanism.

The API does not load the model for every request.

---

## Endpoints

### Health

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0"
}
```

---

### Metadata

```http
GET /metadata
```

Returns model metadata including:

- Model version
- Training artifact timestamp
- Feature names
- Framework
- Model artifact SHA-256 hash

---

### Single prediction

```http
POST /predict
```

Request:

```json
{
  "features": {
    "MSSubClass": 60,
    "MSZoning": "RL",
    "LotArea": 8450,
    "OverallQual": 7,
    "OverallCond": 5,
    "YearBuilt": 2003,
    "YearRemodAdd": 2003,
    "GrLivArea": 1710,
    "FullBath": 2,
    "BedroomAbvGr": 3,
    "KitchenAbvGr": 1,
    "GarageCars": 2,
    "GarageArea": 548
  }
}
```

Example response structure:

```json
{
  "prediction": 197296.21,
  "model_version": "1.0.0",
  "correlation_id": "5d323f87-fc17-495b-8d74-c9091fcc5574",
  "latency_ms": 589.84
}
```

---

### Batch prediction

```http
POST /predict/batch
```

Request:

```json
{
  "requests": [
    {
      "features": {
        "MSSubClass": 60,
        "MSZoning": "RL",
        "LotArea": 8450,
        "OverallQual": 7,
        "OverallCond": 5
      }
    },
    {
      "features": {
        "MSSubClass": 20,
        "MSZoning": "RL",
        "LotArea": 9600,
        "OverallQual": 6,
        "OverallCond": 6
      }
    }
  ]
}
```

---

# 10. API Validation and Error Handling

Pydantic models validate incoming requests.

The API handles:

### Invalid request data

```text
HTTP 422
```

### Unknown feature names

```text
HTTP 422
```

### Unexpected server errors

```text
HTTP 500
```

Error responses contain the request correlation ID so that failures can be traced through the logs.

---

# 11. Testing

Testing is implemented with:

```text
pytest
pytest-cov
```

The test suite covers:

- Feature preprocessing
- Missing values
- Unknown categorical values
- Predictor behavior
- Single predictions
- Batch predictions
- Invalid features
- API endpoints
- API validation errors
- API internal errors
- Structured logging
- Correlation IDs
- Pickle/ONNX prediction parity

The tests also use:

```text
pytest.mark.parametrize
```

and mocking with:

```text
unittest.mock.MagicMock
```

---

## Running tests

Run:

```bash
uv run pytest -v
```

Coverage can be checked using:

```bash
uv run pytest --cov=src/prodml --cov-report=term-missing --cov-fail-under=70
```

Note: the current test suite reached approximately 65% total coverage because training, export, and benchmark modules are executable utility modules that are not fully covered by the current tests. The functional tests for the API, predictor, features, logging, and serialization are passing.

---

# 12. Docker

The application is containerized using a multi-stage Docker build.

The Docker image contains:

- Python runtime
- Production dependencies
- Application source
- Trained model artifact

The training dataset is intentionally excluded from the runtime image because it is not required for inference.

---

## Docker architecture

```text
Builder Stage
     |
     |-- Python
     |-- uv
     |-- production dependencies
     |
     v
Runtime Stage
     |
     |-- Python slim
     |-- .venv
     |-- src/
     |-- models/
     |
     v
FastAPI
```

---

# 13. Build Docker Image

Build the image:

```bash
docker build -t prodml-house-prices:1.0.0 .
```

For a completely fresh build:

```bash
docker build --no-cache -t prodml-house-prices:1.0.0 .
```

---

# 14. Docker Compose

Start the service:

```bash
docker compose up -d
```

Check the container:

```bash
docker compose ps
```

Stop the service:

```bash
docker compose down
```

---

# 15. Health Check

The container uses the FastAPI health endpoint as its Docker health check.

Check the API:

```bash
curl.exe http://localhost:8000/health
```

Expected response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0"
}
```

Check Docker health status:

```bash
docker inspect --format="{{.State.Health.Status}}" prodml-api
```

Expected:

```text
healthy
```

---

# 16. Non-root Container

The application runs as:

```text
appuser
```

instead of root.

Verify:

```bash
docker exec prodml-api whoami
```

Expected:

```text
appuser
```

This reduces the privileges available to the application inside the container.

---

# 17. Swagger Documentation

When the container is running, the interactive API documentation is available at:

```text
http://localhost:8000/docs
```

The OpenAPI schema is available at:

```text
http://localhost:8000/openapi.json
```

---

# 18. Docker Hub

The image can be tagged for Docker Hub using:

```bash
docker tag prodml-house-prices:1.0.0 DOCKERHUB_USERNAME/prodml-house-prices:1.0.0
```

Login:

```bash
docker login
```

Push:

```bash
docker push DOCKERHUB_USERNAME/prodml-house-prices:1.0.0
```

Replace `DOCKERHUB_USERNAME` with the Docker Hub username.

---

# 19. Configuration

Application configuration is defined in `src/prodml/config.py`.

Default configuration includes:

```text
data_path          = data/raw/train.csv
model_path         = models/model.pkl
onnx_model_path    = models/model.onnx
target_column      = SalePrice
validation_size    = 0.2
random_state       = 42
model_version      = 1.0.0
api_host           = 127.0.0.1
api_port           = 8000
```

Configuration can be overridden using environment variables through Pydantic Settings.

---

# 20. Technologies

Main technologies used:

- Python
- pandas
- NumPy
- scikit-learn
- ONNX
- ONNX Runtime
- FastAPI
- Pydantic
- Uvicorn
- pytest
- pytest-cov
- python-json-logger
- joblib
- uv
- Docker
- Docker Compose

---

# 21. MLOps Practices Demonstrated

This project demonstrates the transition from a notebook-based ML workflow to a more production-oriented workflow.

Key practices include:

```text
Notebook
   |
   v
Modular Python package
   |
   v
Reusable training pipeline
   |
   v
Persisted model artifact
   |
   v
Serialization / ONNX
   |
   v
FastAPI inference service
   |
   v
Structured logging
   |
   v
Automated testing
   |
   v
Docker container
   |
   v
Docker Compose
   |
   v
Container health monitoring
```

---

# 22. Project Status

Mini Project 1 has completed the main Module 1 implementation:

- [x] Refactor notebook into Python modules
- [x] Train and save model
- [x] Structured JSON logging
- [x] Pickle serialization
- [x] ONNX export
- [x] Pickle/ONNX prediction parity
- [x] Serialization benchmark
- [x] FastAPI service
- [x] Pydantic request/response schemas
- [x] `/health`
- [x] `/metadata`
- [x] `/predict`
- [x] `/predict/batch`
- [x] Request correlation IDs
- [x] API error handling
- [x] pytest test suite
- [x] Feature and predictor tests
- [x] API tests
- [x] Serialization tests
- [x] Multi-stage Dockerfile
- [x] Docker Compose
- [x] Non-root container user
- [x] Docker health check
- [x] Local container deployment

---

## Author

Mohamed Mo'men

GitHub: `mopharaoh`