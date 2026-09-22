from unittest.mock import MagicMock

import pytest

from prodml.api.main import app


def test_health_endpoint(client):
    """Health endpoint should report a healthy loaded service."""
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "model_version" in data


def test_metadata_endpoint(client):
    """Metadata endpoint should expose model information."""
    response = client.get("/metadata")

    assert response.status_code == 200

    data = response.json()

    assert "model_version" in data
    assert "training_date" in data
    assert "feature_names" in data
    assert "framework" in data
    assert "artifact_hash" in data

    assert data["framework"] == "scikit-learn"
    assert isinstance(data["feature_names"], list)
    assert len(data["feature_names"]) > 0


def test_predict_endpoint(client, sample_features):
    """Predict endpoint should return a valid prediction response."""
    response = client.post(
        "/predict",
        json={"features": sample_features},
    )

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data["prediction"], float)
    assert data["prediction"] > 0
    assert "model_version" in data
    assert "correlation_id" in data
    assert "latency_ms" in data
    assert data["latency_ms"] >= 0

    assert "X-Request-ID" in response.headers
    assert response.headers["X-Request-ID"] == data["correlation_id"]


def test_predict_batch_endpoint(client, sample_features):
    """Batch endpoint should return one prediction per request."""
    response = client.post(
        "/predict/batch",
        json={
            "requests": [
                {"features": sample_features},
                {"features": sample_features},
            ]
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data["predictions"]) == 2
    assert all(
        isinstance(value, float)
        for value in data["predictions"]
    )
    assert "model_version" in data
    assert "correlation_id" in data
    assert data["latency_ms"] >= 0


def test_predict_validation_error(client):
    """Invalid request data should return HTTP 422."""
    response = client.post(
        "/predict",
        json={
            "features": {}
        },
    )

    assert response.status_code == 422

    data = response.json()

    assert data["detail"] == "Invalid request data."
    assert "errors" in data
    assert "correlation_id" in data


def test_predict_missing_features_field(client):
    """Missing required features field should return HTTP 422."""
    response = client.post(
        "/predict",
        json={},
    )

    assert response.status_code == 422

    data = response.json()

    assert data["detail"] == "Invalid request data."
    assert "errors" in data
    assert "correlation_id" in data


def test_predict_unknown_feature(client, sample_features):
    """Unknown feature names should return HTTP 422."""
    invalid_features = {
        **sample_features,
        "UnknownFeature": 123,
    }

    response = client.post(
        "/predict",
        json={"features": invalid_features},
    )

    assert response.status_code == 422

    data = response.json()

    assert "Unknown features" in data["detail"]
    assert "correlation_id" in data


def test_batch_empty_requests(client):
    """An empty batch should be rejected by Pydantic."""
    response = client.post(
        "/predict/batch",
        json={"requests": []},
    )

    assert response.status_code == 422

    data = response.json()

    assert data["detail"] == "Invalid request data."
    assert "errors" in data


def test_batch_too_large(client, sample_features):
    """A batch larger than the schema limit should be rejected."""
    requests = [
        {"features": sample_features}
        for _ in range(1001)
    ]

    response = client.post(
        "/predict/batch",
        json={"requests": requests},
    )

    assert response.status_code == 422


def test_feature_validation_error_handler(
    client,
    sample_features,
):
    """FeatureValidationError should be converted to HTTP 422."""
    predictor = app.state.predictor

    original_method = predictor.predict_one

    try:
        from prodml.predict import FeatureValidationError

        predictor.predict_one = MagicMock(
            side_effect=FeatureValidationError(
                "Test feature validation error"
            )
        )

        response = client.post(
            "/predict",
            json={"features": sample_features},
        )

        assert response.status_code == 422

        data = response.json()

        assert data["detail"] == "Test feature validation error"
        assert "correlation_id" in data

    finally:
        predictor.predict_one = original_method


def test_unexpected_error_handler(
    client,
    sample_features,
):
    """Unexpected predictor errors should return HTTP 500."""
    predictor = app.state.predictor

    original_method = predictor.predict_one

    try:
        predictor.predict_one = MagicMock(
            side_effect=RuntimeError(
                "Simulated internal failure"
            )
        )

        response = client.post(
            "/predict",
            json={"features": sample_features},
        )

        assert response.status_code == 500

        data = response.json()

        assert data["detail"] == "Internal server error."
        assert "correlation_id" in data

    finally:
        predictor.predict_one = original_method


@pytest.mark.parametrize(
    "path",
    [
        "/health",
        "/metadata",
    ],
)
def test_system_endpoints_have_request_id_header(
    client,
    path,
):
    """System endpoints should receive a correlation/request ID."""
    response = client.get(path)

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) > 0