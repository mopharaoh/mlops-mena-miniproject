import hashlib
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prodml.api.schemas import (
    BatchPredictionRequest,
    PredictionRequest,
    BatchPredictionResponse,
    PredictionResponse,
)
from prodml.config import settings
from prodml.data import (
    fill_categorical_missing_values,
    get_categorical_fill_values,
    load_data,
    split_features_target,
    train_validation_split,
)
from prodml.logging_conf import configure_logging

logger = logging.getLogger(__name__)


MODEL_VERSION = "1.0.0"


def calculate_file_hash(path: Path) -> str:
    """Calculate the SHA-256 hash of a model artifact."""

    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def load_api_model() -> dict[str, Any]:
    """Load the model and inference metadata into memory."""

    if not settings.model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {settings.model_path}"
        )

    if not settings.data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {settings.data_path}"
        )

    model = joblib.load(settings.model_path)

    df = load_data(settings.data_path)

    X, y = split_features_target(
        df,
        settings.target_column,
    )

    X_train, _, _, _ = train_validation_split(
        X,
        y,
        validation_size=settings.validation_size,
        random_state=settings.random_state,
    )

    categorical_features = (
        X_train
        .select_dtypes(exclude=["number"])
        .columns
        .tolist()
    )

    fill_values = get_categorical_fill_values(
        X_train,
        categorical_features,
    )

    feature_names = X.columns.tolist()

    artifact_hash = calculate_file_hash(
        settings.model_path
    )

    training_date = pd.Timestamp(
        settings.model_path.stat().st_mtime,
        unit="s",
    ).isoformat()

    return {
        "model": model,
        "fill_values": fill_values,
        "feature_names": feature_names,
        "categorical_features": categorical_features,
        "artifact_hash": artifact_hash,
        "training_date": training_date,
        "framework": "scikit-learn",
        "model_version": MODEL_VERSION,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model once when the application starts."""

    configure_logging()

    logger.info(
        "Loading model at application startup",
        extra={
            "model_path": str(settings.model_path),
        },
    )

    app.state.model_data = load_api_model()

    logger.info(
        "Model loaded successfully",
        extra={
            "model_version": app.state.model_data["model_version"],
            "framework": app.state.model_data["framework"],
        },
    )

    yield

    logger.info("Application shutdown")


app = FastAPI(
    title="House Prices Prediction API",
    description=(
        "Production-ready FastAPI service for predicting "
        "House Prices using a trained scikit-learn model."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)


@app.middleware("http")
async def correlation_id_middleware(
    request: Request,
    call_next,
):
    """Attach a UUID4 correlation ID to every request."""

    correlation_id = str(uuid.uuid4())

    request.state.correlation_id = correlation_id

    response = await call_next(request)

    response.headers["X-Request-ID"] = correlation_id

    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    """Return a clean 422 response for invalid requests."""

    correlation_id = getattr(
        request.state,
        "correlation_id",
        "unknown",
    )

    logger.warning(
        "Request validation failed",
        extra={
            "correlation_id": correlation_id,
            "errors": exc.errors(),
        },
    )

    return JSONResponse(
        status_code=422,
        content={
            "detail": "Invalid request data.",
            "errors": exc.errors(),
            "correlation_id": correlation_id,
        },
    )


@app.exception_handler(Exception)
async def unexpected_exception_handler(
    request: Request,
    exc: Exception,
):
    """Return a clean 500 response without leaking internals."""

    correlation_id = getattr(
        request.state,
        "correlation_id",
        "unknown",
    )

    logger.exception(
        "Unexpected API error",
        extra={
            "correlation_id": correlation_id,
        },
        exc_info=exc,
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error.",
            "correlation_id": correlation_id,
        },
    )


@app.get(
    "/health",
    tags=["system"],
)
async def health(request: Request) -> dict[str, Any]:
    """
    Check whether the ML model is loaded in memory.
    """

    model_data = getattr(
        request.app.state,
        "model_data",
        None,
    )

    if model_data is None or model_data.get("model") is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "detail": "Model is not loaded.",
            },
        )

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": model_data["model_version"],
    }


@app.get(
    "/metadata",
    tags=["system"],
)
async def metadata(request: Request) -> dict[str, Any]:
    """Return metadata about the loaded model."""

    model_data = request.app.state.model_data

    return {
        "model_version": model_data["model_version"],
        "training_date": model_data["training_date"],
        "feature_names": model_data["feature_names"],
        "framework": model_data["framework"],
        "artifact_hash": model_data["artifact_hash"],
    }


@app.post(
    "/predict",
    tags=["prediction"],
)
async def predict(
    payload: PredictionRequest,
    request: Request,
) -> dict[str, Any]:
    """Generate a prediction for a single house."""


    correlation_id = request.state.correlation_id

    start_time = time.perf_counter()

    model_data = request.app.state.model_data

    model = model_data["model"]
    fill_values = model_data["fill_values"]
    feature_names = model_data["feature_names"]

    input_features = payload.features

    missing_features = [
        feature
        for feature in feature_names
        if feature not in input_features
    ]

    if missing_features:
        raise ValueError(
            "Missing required features: "
            + ", ".join(missing_features)
        )

    extra_features = [
        feature
        for feature in input_features
        if feature not in feature_names
    ]

    if extra_features:
        raise ValueError(
            "Unknown features: "
            + ", ".join(extra_features)
        )

    features_df = pd.DataFrame(
        [input_features],
        columns=feature_names,
    )

    features_df = fill_categorical_missing_values(
        features_df,
        model_data["categorical_features"],
        fill_values,
    )

    prediction = float(
        model.predict(features_df)[0]
    )

    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    logger.info(
        "Prediction served",
        extra={
            "correlation_id": correlation_id,
            "latency_ms": latency_ms,
        },
    )

    response = PredictionResponse(
        prediction=prediction,
        model_version=model_data["model_version"],
        correlation_id=correlation_id,
        latency_ms=latency_ms,
    )

    return response.model_dump()


@app.post(
    "/predict/batch",
    tags=["prediction"],
)
async def predict_batch(
    payload: BatchPredictionRequest,
    request: Request,
) -> dict[str, Any]:
    """Generate predictions for multiple houses."""


    correlation_id = request.state.correlation_id

    start_time = time.perf_counter()

    model_data = request.app.state.model_data

    model = model_data["model"]
    fill_values = model_data["fill_values"]
    feature_names = model_data["feature_names"]

    rows = [
        item.features
        for item in payload.requests
    ]

    for index, row in enumerate(rows):
        missing_features = [
            feature
            for feature in feature_names
            if feature not in row
        ]

        if missing_features:
            raise ValueError(
                f"Request {index} is missing required features: "
                + ", ".join(missing_features)
            )

        extra_features = [
            feature
            for feature in row
            if feature not in feature_names
        ]

        if extra_features:
            raise ValueError(
                f"Request {index} contains unknown features: "
                + ", ".join(extra_features)
            )

    features_df = pd.DataFrame(
        rows,
        columns=feature_names,
    )

    features_df = fill_categorical_missing_values(
        features_df,
        model_data["categorical_features"],
        fill_values,
    )

    predictions = [
        float(value)
        for value in model.predict(features_df)
    ]

    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    logger.info(
        "Batch prediction served",
        extra={
            "correlation_id": correlation_id,
            "latency_ms": latency_ms,
            "batch_size": len(rows),
        },
    )

    response = BatchPredictionResponse(
        predictions=predictions,
        model_version=model_data["model_version"],
        correlation_id=correlation_id,
        latency_ms=latency_ms,
    )

    return response.model_dump()
