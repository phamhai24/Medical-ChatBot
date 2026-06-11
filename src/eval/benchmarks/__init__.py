"""Benchmarks package."""

from src.eval.benchmarks.medical_qa import (
    get_benchmark_dataset,
    get_benchmark_by_category,
    get_benchmark_by_difficulty,
    BENCHMARK_DATASET,
)

__all__ = [
    "get_benchmark_dataset",
    "get_benchmark_by_category",
    "get_benchmark_by_difficulty",
    "BENCHMARK_DATASET",
]
