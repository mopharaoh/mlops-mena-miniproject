import joblib
import pandas as pd

from prodml.config import settings
from prodml.data import (
    fill_categorical_missing_values,
    get_categorical_fill_values,
    load_data,
    split_features_target,
    train_validation_split,
)
from prodml.export import export_to_onnx


def main() -> None:
    """Export the trained model to ONNX."""

    df = load_data(
        settings.data_path
    )

    X, _ = split_features_target(
        df,
        settings.target_column,
    )

    X_train, X_val, _, _ = train_validation_split(
        X,
        df[settings.target_column],
        validation_size=settings.validation_size,
        random_state=settings.random_state,
    )

    categorical_features = (
        X_train
        .select_dtypes(
            exclude=["number"]
        )
        .columns
        .tolist()
    )

    fill_values = get_categorical_fill_values(
        X_train,
        categorical_features,
    )

    X_val = fill_categorical_missing_values(
        X_val,
        categorical_features,
        fill_values,
    )

    model = joblib.load(
        settings.model_path
    )

    export_to_onnx(
        model=model,
        sample_features=X_val.iloc[:1],
        output_path=settings.onnx_model_path,
    )

    print(
        f"ONNX model saved to: "
        f"{settings.onnx_model_path}"
    )


if __name__ == "__main__":
    main()