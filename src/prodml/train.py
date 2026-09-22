import logging
from pathlib import Path

import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline

from prodml.config import settings
from prodml.data import (
    load_data,
    split_features_target,
    train_validation_split,
)
from prodml.features import (
    build_preprocessor,
    fill_categorical_missing_values,
    get_categorical_fill_values,
)
from prodml.logging_conf import configure_logging
from prodml.predict import HousePricePredictor

logger = logging.getLogger(__name__)


def train_model(
    data_path: Path,
    model_path: Path,
    target_column: str,
    validation_size: float,
    random_state: int,
    model_version: str,
) -> dict[str, float]:
    """Train and persist the complete House Prices predictor."""

    df = load_data(
        data_path
    )

    X, y = split_features_target(
        df,
        target_column,
    )

    X_train, X_val, y_train, y_val = (
        train_validation_split(
            X,
            y,
            validation_size=validation_size,
            random_state=random_state,
        )
    )

    numeric_features = (
        X_train
        .select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    categorical_features = (
        X_train
        .select_dtypes(
            exclude=["number"]
        )
        .columns
        .tolist()
    )

    # Learn categorical fill values from training data only.
    categorical_fill_values = (
        get_categorical_fill_values(
            X_train,
            categorical_features,
        )
    )

    # Apply the learned values to both training and validation data.
    X_train = (
        fill_categorical_missing_values(
            X_train,
            categorical_features,
            categorical_fill_values,
        )
    )

    X_val = (
        fill_categorical_missing_values(
            X_val,
            categorical_features,
            categorical_fill_values,
        )
    )

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "regressor",
                LinearRegression(),
            ),
        ]
    )

    logger.info(
        "Model training started",
        extra={
            "model": "LinearRegression",
            "training_rows": len(X_train),
            "validation_rows": len(X_val),
        },
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_val
    )

    mae = mean_absolute_error(
        y_val,
        predictions,
    )

    rmse = mean_squared_error(
        y_val,
        predictions,
    ) ** 0.5

    predictor = HousePricePredictor(
        model=model,
        categorical_fill_values=(
            categorical_fill_values
        ),
        model_version=model_version,
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save the complete predictor artifact.
    # This contains both the model and the learned
    # categorical preprocessing information.
    joblib.dump(
        predictor,
        model_path,
    )

    logger.info(
        "Model artifact saved",
        extra={
            "model_path": str(
                model_path
            ),
        },
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
    }


def main() -> None:
    """Run model training."""

    configure_logging()

    metrics = train_model(
        data_path=settings.data_path,
        model_path=settings.model_path,
        target_column=settings.target_column,
        validation_size=settings.validation_size,
        random_state=settings.random_state,
        model_version=settings.model_version,
    )

    logger.info(
        "Model training completed",
        extra={
            "model": "LinearRegression",
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "model_path": str(
                settings.model_path
            ),
        },
    )


if __name__ == "__main__":
    main()