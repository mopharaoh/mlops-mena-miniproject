import time
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any
import logging
import joblib
import pandas as pd

logger = logging.getLogger(__name__)

def timed(func: Callable[..., Any]) -> Callable[..., Any]:
    """Measure and print function execution time."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()

        result = func(*args, **kwargs)

        elapsed = time.perf_counter() - start

        logger.info(
            "Function execution completed",
            extra={
                "function": func.__name__,
                "latency_ms": elapsed * 1000,
            },
        )

        return result

    return wrapper


class HousePricePredictor:
    """Load a trained House Prices model and make predictions."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.model = None

    def load(self) -> None:
        

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        self.model = joblib.load(self.model_path)

    @timed
    def predict_one(
        self,
        features: pd.DataFrame,
    ) -> float:
        """Predict the house price for one sample."""

        if self.model is None:
            raise RuntimeError(
                "Model is not loaded. Call load() first."
            )

        prediction = self.model.predict(features)

        return float(prediction[0])

    def predict_batch(
        self,
        features: pd.DataFrame,
    ) -> list[float]:
        """Predict house prices for multiple samples."""

        if self.model is None:
            raise RuntimeError(
                "Model is not loaded. Call load() first."
            )

        predictions = self.model.predict(features)

        return predictions.tolist()