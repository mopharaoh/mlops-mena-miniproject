import numpy as np
import onnxruntime as ort

from prodml.config import settings
from prodml.data import (
    load_data,
    split_features_target,
    train_validation_split,
)
from prodml.predict import HousePricePredictor


def resolve_onnx_column(
    input_name: str,
    features,
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
        f"ONNX input '{input_name}' "
        "does not match any dataframe column."
    )


def test_pickle_and_onnx_prediction_parity() -> None:
    """Ensure Pickle and ONNX predictions are equivalent."""

    assert settings.model_path.exists()
    assert settings.onnx_model_path.exists()

    df = load_data(
        settings.data_path
    )

    X, y = split_features_target(
        df,
        settings.target_column,
    )

    _, X_val, _, _ = train_validation_split(
        X,
        y,
        validation_size=settings.validation_size,
        random_state=settings.random_state,
    )

    predictor = HousePricePredictor.load(
        settings.model_path,
        settings.model_version,
    )

    # IMPORTANT:
    # Both Pickle and ONNX must receive
    # exactly the same prepared features.
    prepared = predictor.prepare_for_onnx(
        X_val
    )

    pred_pickle = predictor.model.predict(
        prepared
    )

    session = ort.InferenceSession(
        str(settings.onnx_model_path),
        providers=[
            "CPUExecutionProvider"
        ],
    )

    inputs = {}

    for input_meta in session.get_inputs():
        input_name = input_meta.name

        column = resolve_onnx_column(
            input_name,
            prepared,
        )

        values = prepared[
            column
        ].to_numpy()

        if input_meta.type == "tensor(float)":
            values = values.astype(
                np.float32
            )

        elif input_meta.type == "tensor(double)":
            values = values.astype(
                np.float64
            )

        elif input_meta.type == "tensor(string)":
            values = values.astype(str)

        inputs[input_name] = (
            values.reshape(-1, 1)
        )

    outputs = session.run(
        None,
        inputs,
    )

    pred_onnx = outputs[0].reshape(-1)

    assert len(pred_pickle) == len(
        pred_onnx
    )

    assert np.allclose(
        pred_pickle,
        pred_onnx,
        atol=1e-4,
    )