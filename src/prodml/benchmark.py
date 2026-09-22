import json
import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np
import onnxruntime as ort

from prodml.config import settings
from prodml.data import (
    load_data,
    split_features_target,
    train_validation_split,
)
from prodml.logging_conf import configure_logging
from prodml.predict import HousePricePredictor

logger = logging.getLogger(__name__)
import pandas as pd


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

def benchmark(
    model_fn: Callable[[], object],
    repetitions: int = 100,
    warmup: int = 20,
) -> dict[str, float]:
    """Benchmark a callable and return mean/p95 latency."""

    for _ in range(warmup):
        model_fn()

    latencies = []

    for _ in range(repetitions):
        start = time.perf_counter()

        model_fn()

        end = time.perf_counter()

        latencies.append(
            (end - start) * 1000
        )

    values = np.asarray(
        latencies
    )

    return {
        "mean_ms": float(
            np.mean(values)
        ),
        "p95_ms": float(
            np.percentile(
                values,
                95,
            )
        ),
    }


def create_onnx_inputs(
    session: ort.InferenceSession,
    features,
) -> dict[str, np.ndarray]:
    """Create ONNX Runtime inputs."""

    inputs = {}

    

    for input_meta in session.get_inputs():
        input_name = input_meta.name

        column = resolve_onnx_column(input_name, features)
        values = features[
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

    return inputs


def main() -> None:
    """Benchmark Pickle and ONNX."""

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

    prepared = predictor.prepare_for_onnx(
        X_val
    )

    session = ort.InferenceSession(
        str(settings.onnx_model_path),
        providers=[
            "CPUExecutionProvider"
        ],
    )

    onnx_inputs = create_onnx_inputs(
        session,
        prepared,
    )

    pickle_result = benchmark(
        lambda: predictor.model.predict(
            X_val
        )
    )

    onnx_result = benchmark(
        lambda: session.run(
            None,
            onnx_inputs,
        )
    )

    results = {
        "dataset_rows": len(X_val),
        "repetitions": 100,
        "warmup": 20,
        "pickle": pickle_result,
        "onnx": onnx_result,
    }

    output_path = Path(
        "reports/serialization_benchmark.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info(
        "Serialization benchmark completed",
        extra={
            "pickle_mean_ms": pickle_result[
                "mean_ms"
            ],
            "pickle_p95_ms": pickle_result[
                "p95_ms"
            ],
            "onnx_mean_ms": onnx_result[
                "mean_ms"
            ],
            "onnx_p95_ms": onnx_result[
                "p95_ms"
            ],
        },
    )


if __name__ == "__main__":
    main()