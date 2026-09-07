import time
from typing import Callable

import numpy as np


def benchmark(
    model_fn: Callable[[], object],
    repetitions: int = 100,
    warmup: int = 20,
) -> dict[str, float]:
    """
    Benchmark a prediction function.

    Returns mean and p95 latency in milliseconds.
    """

    for _ in range(warmup):
        model_fn()

    latencies_ms: list[float] = []

    for _ in range(repetitions):
        start = time.perf_counter()

        model_fn()

        end = time.perf_counter()

        latency_ms = (end - start) * 1000

        latencies_ms.append(latency_ms)

    latency_array = np.array(latencies_ms)

    return {
        "mean_ms": float(np.mean(latency_array)),
        "p95_ms": float(np.percentile(latency_array, 95)),
    }