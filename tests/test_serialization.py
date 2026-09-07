import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

from prodml.config import settings
from prodml.data import (
    fill_categorical_missing_values,
    get_categorical_fill_values,
    load_data,
    split_features_target,
    train_validation_split,
)


def load_validation_features() -> pd.DataFrame:
    """Load the same validation features used during training."""

    df = load_data(settings.data_path)

    X, y = split_features_target(
        df,
        settings.target_column,
    )

    X_train, X_val, _, _ = train_validation_split(
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

    X_val = fill_categorical_missing_values(
        X_val,
        categorical_features,
        fill_values,
    )

    return X_val.copy()


def resolve_onnx_column(
    input_name: str,
    features: pd.DataFrame,
) -> str:
    """Resolve an ONNX input name to the original dataframe column."""

    if input_name in features.columns:
        return input_name

    candidate = input_name

    while candidate.startswith("_"):
        candidate = candidate[1:]

        if candidate in features.columns:
            return candidate

    raise KeyError(
        f"ONNX input '{input_name}' does not match any dataframe column."
    )


def create_onnx_inputs(
    session: ort.InferenceSession,
    features: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Convert dataframe columns to ONNX inputs."""

    inputs: dict[str, np.ndarray] = {}

    for input_meta in session.get_inputs():
        input_name = input_meta.name

        dataframe_column = resolve_onnx_column(
            input_name,
            features,
        )

        values = features[dataframe_column].to_numpy()

        if input_meta.type == "tensor(float)":
            values = values.astype(np.float32)

        elif input_meta.type == "tensor(double)":
            values = values.astype(np.float64)

        elif input_meta.type == "tensor(string)":
            values = values.astype(str)

        inputs[input_name] = values.reshape(-1, 1)

    return inputs


def test_pickle_and_onnx_prediction_parity() -> None:
    """Ensure Pickle and ONNX predictions are equivalent."""

    assert settings.model_path.exists()
    assert settings.onnx_model_path.exists()

    X_val = load_validation_features()

    # House Prices validation split contains 292 rows.
    assert len(X_val) == 292

    pickle_model = joblib.load(settings.model_path)

    pred_pickle = pickle_model.predict(X_val)

    session = ort.InferenceSession(
        str(settings.onnx_model_path),
        providers=["CPUExecutionProvider"],
    )

    onnx_inputs = create_onnx_inputs(
        session,
        X_val,
    )

    onnx_outputs = session.run(
        None,
        onnx_inputs,
    )

    pred_onnx = onnx_outputs[0].reshape(-1)

    assert len(pred_pickle) == 292
    assert len(pred_onnx) == 292

    assert np.allclose(
        pred_pickle,
        pred_onnx,
        atol=1e-4,
    )