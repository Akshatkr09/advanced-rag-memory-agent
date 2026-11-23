"""Latency timing helper."""
import time
from contextlib import contextmanager

@contextmanager
def timer():
    start = time.perf_counter()
    result = {}
    try:
        yield result
    finally:
        result["latency_ms"] = max((time.perf_counter() - start) * 1000.0, 0.001)
