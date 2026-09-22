import hashlib
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from prodml.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from prodml.config import settings
from prodml.logging_conf import (
    configure_logging,
    reset_correlation_id,
    set_correlation_id,
)
from prodml.predict import (
    FeatureValidationError,
    HousePricePredictor,
)

logger = logging.getLogger(__name__)


def calculate_file_hash(
    path: Path,
) -> str:
    """Calculate the SHA-256 hash of a file."""

    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(8192),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


def load_predictor() -> HousePricePredictor:
    """Load the predictor once at application startup."""

    return HousePricePredictor.load(
        model_path=settings.model_path,
        model_version=settings.model_version,
    )


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    """Application startup and shutdown lifecycle."""

    configure_logging()

    logger.info(
        "Loading model at application startup",
        extra={
            "model_path": str(
                settings.model_path
            )
        },
    )

    app.state.predictor = load_predictor()

    logger.info(
        "Model loaded successfully",
        extra={
            "model_version": (
                app.state.predictor.model_version
            ),
            "framework": "scikit-learn",
        },
    )

    yield

    logger.info(
        "Application shutdown"
    )


app = FastAPI(
    title="House Prices Prediction API",
    description=(
        "Production-ready API for House Prices "
        "prediction using scikit-learn."
    ),
    version=settings.model_version,
    lifespan=lifespan,
)


@app.middleware("http")
async def correlation_id_middleware(
    request: Request,
    call_next,
):
    """Attach a UUID4 correlation ID to every request."""

    correlation_id = str(
        uuid.uuid4()
    )

    request.state.correlation_id = (
        correlation_id
    )

    token = set_correlation_id(
        correlation_id
    )

    start_time = time.perf_counter()

    try:
        response = await call_next(
            request
        )

        response.headers[
            "X-Request-ID"
        ] = correlation_id

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        logger.info(
            "Request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
            },
        )

        return response

    finally:
        reset_correlation_id(
            token
        )


@app.exception_handler(
    RequestValidationError
)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    """Return clean 422 validation responses."""

    correlation_id = getattr(
        request.state,
        "correlation_id",
        "unknown",
    )

    logger.warning(
        "Request validation failed",
        extra={
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


@app.exception_handler(
    FeatureValidationError
)
async def feature_validation_exception_handler(
    request: Request,
    exc: FeatureValidationError,
):
    """Return clean 422 feature-name errors."""

    correlation_id = getattr(
        request.state,
        "correlation_id",
        "unknown",
    )

    logger.warning(
        "Feature validation failed",
        extra={
            "error": str(exc),
        },
    )

    return JSONResponse(
        status_code=422,
        content={
            "detail": str(exc),
            "correlation_id": correlation_id,
        },
    )


@app.exception_handler(
    Exception
)
async def unexpected_exception_handler(
    request: Request,
    exc: Exception,
):
    """
    Handle unexpected errors without leaking
    implementation details to the client.
    """

    correlation_id = getattr(
        request.state,
        "correlation_id",
        "unknown",
    )

    logger.exception(
        "Unexpected API error",
        extra={},
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
async def health(
    request: Request,
) -> dict[str, Any]:
    """Return service health."""

    predictor = getattr(
        request.app.state,
        "predictor",
        None,
    )

    if predictor is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "model_loaded": False,
            },
        )

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": (
            predictor.model_version
        ),
    }


@app.get(
    "/metadata",
    tags=["system"],
)
async def metadata(
    request: Request,
) -> dict[str, Any]:
    """Return model metadata."""

    predictor: HousePricePredictor = (
        request.app.state.predictor
    )

    return {
        "model_version": (
            predictor.model_version
        ),
        "training_date": (
            settings.model_path
            .stat()
            .st_mtime
        ),
        "feature_names": (
            predictor.feature_names
        ),
        "framework": "scikit-learn",
        "artifact_hash": calculate_file_hash(
            settings.model_path
        ),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["prediction"],
)
async def predict(
    payload: PredictionRequest,
    request: Request,
) -> PredictionResponse:
    """Generate a single house-price prediction."""

    predictor: HousePricePredictor = (
        request.app.state.predictor
    )

    correlation_id = (
        request.state.correlation_id
    )

    start_time = time.perf_counter()

    prediction = predictor.predict_one(
        payload.features
    )

    latency_ms = (
        time.perf_counter()
        - start_time
    ) * 1000

    logger.info(
        "Prediction served",
        extra={
            "latency_ms": latency_ms,
        },
    )

    return PredictionResponse(
        prediction=prediction,
        model_version=(
            predictor.model_version
        ),
        correlation_id=correlation_id,
        latency_ms=latency_ms,
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["prediction"],
)
async def predict_batch(
    payload: BatchPredictionRequest,
    request: Request,
) -> BatchPredictionResponse:
    """Generate batch house-price predictions."""

    predictor: HousePricePredictor = (
        request.app.state.predictor
    )

    correlation_id = (
        request.state.correlation_id
    )

    start_time = time.perf_counter()

    features = [
        item.features
        for item in payload.requests
    ]

    predictions = predictor.predict_batch(
        features
    )

    latency_ms = (
        time.perf_counter()
        - start_time
    ) * 1000

    logger.info(
        "Batch prediction served",
        extra={
            "latency_ms": latency_ms,
            "batch_size": len(features),
        },
    )

    return BatchPredictionResponse(
        predictions=predictions,
        model_version=(
            predictor.model_version
        ),
        correlation_id=correlation_id,
        latency_ms=latency_ms,
    )