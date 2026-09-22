from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

from prodml.api.main import app
from prodml.config import settings
from prodml.predict import HousePricePredictor


@pytest.fixture
def sample_features() -> dict:
    """Return a valid sample feature set for API and predictor tests."""
    return {
        "MSSubClass": 60,
        "MSZoning": "RL",
        "LotArea": 8450,
        "Street": "Pave",
        "LotShape": "Reg",
        "LandContour": "Lvl",
        "Utilities": "AllPub",
        "LotConfig": "Inside",
        "LandSlope": "Gtl",
        "Neighborhood": "CollgCr",
        "Condition1": "Norm",
        "Condition2": "Norm",
        "BldgType": "1Fam",
        "HouseStyle": "2Story",
        "OverallQual": 7,
        "OverallCond": 5,
        "YearBuilt": 2003,
        "YearRemodAdd": 2003,
        "TotalBsmtSF": 856,
        "GrLivArea": 1710,
        "FullBath": 2,
        "HalfBath": 1,
        "BedroomAbvGr": 3,
        "KitchenAbvGr": 1,
        "TotRmsAbvGrd": 8,
        "Fireplaces": 0,
        "GarageCars": 2,
        "GarageArea": 548,
        "WoodDeckSF": 0,
        "OpenPorchSF": 61,
        "EnclosedPorch": 0,
        "3SsnPorch": 0,
        "ScreenPorch": 0,
        "PoolArea": 0,
        "MiscVal": 0,
        "MoSold": 2,
        "YrSold": 2008,
    }


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