import logging
from pathlib import Path

import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    FloatTensorType,
    StringTensorType,
)

from prodml.config import settings
from prodml.data import (
    load_data,
    split_features_target,
    train_validation_split,
)
from prodml.logging_conf import configure_logging
from prodml.predict import HousePricePredictor

logger = logging.getLogger(__name__)


def export_to_onnx(
    predictor: HousePricePredictor,
    sample_features: pd.DataFrame,
    output_path: Path,
) -> None:
    """Export the fitted sklearn pipeline to ONNX."""

    prepared = predictor.prepare_for_onnx(
        sample_features
    )

    initial_types = []

    for column in predictor.feature_names:
        if pd.api.types.is_numeric_dtype(
            prepared[column]
        ):
            initial_types.append(
                (
                    column,
                    FloatTensorType(
                        [None, 1]
                    ),
                )
            )
        else:
            initial_types.append(
                (
                    column,
                    StringTensorType(
                        [None, 1]
                    ),
                )
            )

    onnx_model = convert_sklearn(
        predictor.model,
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

    logger.info(
        "ONNX model exported",
        extra={
            "output_path": str(
                output_path
            ),
        },
    )


def main() -> None:
    """Export the trained model to ONNX."""

    configure_logging()

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

    export_to_onnx(
        predictor=predictor,
        sample_features=X_val.iloc[:1],
        output_path=settings.onnx_model_path,
    )


if __name__ == "__main__":
    main()