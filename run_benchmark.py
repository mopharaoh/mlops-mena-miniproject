import joblib
import json

import onnxruntime as ort

from prodml.benchmark import benchmark
from prodml.config import settings
from tests.test_serialization import (
    create_onnx_inputs,
    load_validation_features,
)


def main() -> None:
    """Benchmark Pickle and ONNX inference."""

    X_val = load_validation_features()

    print(
        f"Benchmark dataset: {len(X_val)} validation rows"
    )

    # ---------------------------------------------------------
    # Pickle
    # ---------------------------------------------------------

    pickle_model = joblib.load(
        settings.model_path
    )

    pickle_result = benchmark(
        model_fn=lambda: pickle_model.predict(X_val),
        repetitions=100,
        warmup=20,
    )

    # ---------------------------------------------------------
    # ONNX Runtime
    # ---------------------------------------------------------

    session = ort.InferenceSession(
        str(settings.onnx_model_path),
        providers=["CPUExecutionProvider"],
    )

    onnx_inputs = create_onnx_inputs(
        session,
        X_val,
    )

    onnx_result = benchmark(
        model_fn=lambda: session.run(
            None,
            onnx_inputs,
        ),
        repetitions=100,
        warmup=20,
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    results = {
        "dataset_rows": len(X_val),
        "repetitions": 100,
        "warmup": 20,
        "pickle": pickle_result,
        "onnx": onnx_result,
    }

    print()
    print("=" * 60)
    print("SERIALIZATION BENCHMARK")
    print("=" * 60)

    print()
    print("Pickle / scikit-learn")
    print(
        f"Mean latency : {pickle_result['mean_ms']:.4f} ms"
    )
    print(
        f"P95 latency  : {pickle_result['p95_ms']:.4f} ms"
    )

    print()
    print("ONNX Runtime")
    print(
        f"Mean latency : {onnx_result['mean_ms']:.4f} ms"
    )
    print(
        f"P95 latency  : {onnx_result['p95_ms']:.4f} ms"
    )

    mean_speedup = (
        pickle_result["mean_ms"]
        / onnx_result["mean_ms"]
    )

    p95_speedup = (
        pickle_result["p95_ms"]
        / onnx_result["p95_ms"]
    )

    print()
    print("Speedup")
    print(
        f"Mean speedup : {mean_speedup:.2f}x"
    )
    print(
        f"P95 speedup  : {p95_speedup:.2f}x"
    )

    print()
    print("JSON:")
    print(
        json.dumps(
            results,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()