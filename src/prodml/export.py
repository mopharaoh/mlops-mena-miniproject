from pathlib import Path

import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    FloatTensorType,
    StringTensorType,
)


def build_initial_types(
    features: pd.DataFrame,
) -> list[tuple[str, object]]:
    """Build ONNX input definitions from dataframe columns."""

    initial_types: list[tuple[str, object]] = []

    for column in features.columns:
        if pd.api.types.is_numeric_dtype(
            features[column]
        ):
            initial_types.append(
                (
                    column,
                    FloatTensorType([None, 1]),
                )
            )
        else:
            initial_types.append(
                (
                    column,
                    StringTensorType([None, 1]),
                )
            )

    return initial_types


def export_to_onnx(
    model,
    sample_features: pd.DataFrame,
    output_path: Path,
) -> None:
    """Export a fitted scikit-learn pipeline to ONNX."""

    initial_types = build_initial_types(
        sample_features
    )

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_types,
        target_opset=17,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_bytes(
        onnx_model.SerializeToString()
    )