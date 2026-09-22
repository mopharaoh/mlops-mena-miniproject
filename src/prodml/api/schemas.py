from typing import Any

from pydantic import BaseModel, ConfigDict, Field


EXAMPLE_FEATURES = {
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


class PredictionRequest(BaseModel):
    """Request schema for a single prediction."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "features": EXAMPLE_FEATURES
            }
        }
    )

    features: dict[str, Any] = Field(
        ...,
        min_length=1,
        max_length=78,
        description=(
            "House Prices feature values. "
            "Missing features are automatically "
            "imputed by the trained pipeline."
        ),
    )


class PredictionResponse(BaseModel):
    """Response schema for a single prediction."""

    prediction: float = Field(
        ...,
        description="Predicted house sale price.",
    )

    model_version: str = Field(
        ...,
        description="Version of the model used.",
    )

    correlation_id: str = Field(
        ...,
        description="Request correlation identifier.",
    )

    latency_ms: float = Field(
        ...,
        ge=0,
        description="Prediction latency in milliseconds.",
    )


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "requests": [
                    {
                        "features": EXAMPLE_FEATURES
                    }
                ]
            }
        }
    )

    requests: list[PredictionRequest] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of prediction requests.",
    )


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""

    predictions: list[float] = Field(
        ...,
        description="Predicted house sale prices.",
    )

    model_version: str = Field(
        ...,
        description="Version of the model used.",
    )

    correlation_id: str = Field(
        ...,
        description="Request correlation identifier.",
    )

    latency_ms: float = Field(
        ...,
        ge=0,
        description="Batch prediction latency in milliseconds.",
    )