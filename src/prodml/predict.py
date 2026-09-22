import logging
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from prodml.features import (
    fill_categorical_missing_values,
)

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class FeatureValidationError(ValueError):
    """Raised when an input contains unknown feature names."""


def timed(func: F) -> F:
    """Measure execution time for a function."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()

        try:
            return func(*args, **kwargs)
        finally:
            latency_ms = (
                time.perf_counter() - start
            ) * 1000

            logger.debug(
                "Prediction method completed",
                extra={
                    "latency_ms": latency_ms,
                },
            )

    return wrapper  # type: ignore[return-value]


@dataclass
class HousePricePredictor:
    """
    Production inference interface.

    The predictor owns:
    - the fitted sklearn pipeline
    - categorical fill values
    - model version

    The API does not need access to the training dataset.
    """

    model: Pipeline
    categorical_fill_values: dict[str, str]
    model_version: str = "1.0.0"

    @classmethod
    def load(
        cls,
        model_path: Path,
        model_version: str = "1.0.0",
    ) -> "HousePricePredictor":
        """Load the complete predictor artifact."""

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {model_path}"
            )

        predictor = joblib.load(
            model_path
        )

        if not isinstance(
            predictor,
            cls,
        ):
            raise TypeError(
                "The model artifact must contain "
                "a HousePricePredictor."
            )

        return predictor

    @property
    def feature_names(self) -> list[str]:
        """Return features expected by the model."""

        if not hasattr(
            self.model,
            "feature_names_in_",
        ):
            raise AttributeError(
                "The trained model does not expose "
                "feature_names_in_."
            )

        return list(
            self.model.feature_names_in_
        )

    @property
    def categorical_features(self) -> list[str]:
        """Return categorical feature names."""

        return list(
            self.categorical_fill_values.keys()
        )

    def _validate_features(
        self,
        features: dict[str, Any],
    ) -> None:
        """Validate feature names."""

        expected = set(
            self.feature_names
        )

        received = set(features)

        unknown_features = sorted(
            received - expected
        )

        if unknown_features:
            raise FeatureValidationError(
                "Unknown features: "
                + ", ".join(
                    unknown_features
                )
            )

    def _to_dataframe(
        self,
        features: dict[str, Any],
    ) -> pd.DataFrame:
        """Convert request features to a model dataframe."""

        self._validate_features(
            features
        )

        dataframe = pd.DataFrame(
            [features],
            columns=self.feature_names,
        )

        dataframe = (
            fill_categorical_missing_values(
                dataframe,
                self.categorical_features,
                self.categorical_fill_values,
            )
        )

        return dataframe

    @timed
    def predict_one(
        self,
        features: dict[str, Any],
    ) -> float:
        """Generate one prediction."""

        features_df = self._to_dataframe(
            features
        )

        prediction = self.model.predict(
            features_df
        )

        return float(
            prediction[0]
        )

    @timed
    def predict_batch(
        self,
        features: list[dict[str, Any]],
    ) -> list[float]:
        """Generate predictions for multiple inputs."""

        if not features:
            raise ValueError(
                "At least one feature set is required."
            )

        frames = [
            self._to_dataframe(
                row
            )
            for row in features
        ]

        features_df = pd.concat(
            frames,
            ignore_index=True,
        )

        predictions = self.model.predict(
            features_df
        )

        return [
            float(value)
            for value in np.asarray(
                predictions
            ).reshape(-1)
        ]

    def prepare_for_onnx(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Prepare features for ONNX Runtime.

        The same categorical fill values used during
        API inference are applied here.
        """

        prepared = features.copy()

        prepared = (
            fill_categorical_missing_values(
                prepared,
                self.categorical_features,
                self.categorical_fill_values,
            )
        )

        return prepared[
            self.feature_names
        ].copy()