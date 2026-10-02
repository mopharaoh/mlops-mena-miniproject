import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from prodml.api.main import app
from prodml.config import settings
from prodml.predict import HousePricePredictor


@pytest.fixture
def sample_features() -> dict:
    """Return a valid sample feature set for API and predictor tests."""
    path = Path(__file__).parent / "fixtures" / "house.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def trained_model() -> HousePricePredictor:
    """Load the trained predictor once for the whole test session."""
    return HousePricePredictor.load(
        model_path=settings.model_path,
        model_version=settings.model_version,
    )


@pytest.fixture
def client() -> TestClient:
    """Create a FastAPI test client with application lifespan enabled."""
    with TestClient(
        app,
        raise_server_exceptions=False,
    ) as test_client:
        yield test_client
