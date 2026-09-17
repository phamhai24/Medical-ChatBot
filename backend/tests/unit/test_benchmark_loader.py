"""Tests for benchmark dataset loading."""

from src.eval.benchmarks.medical_qa import get_benchmark_dataset


def test_loads_utf8_json_benchmark():
    """The active benchmark should come from the UTF-8 JSON file when present."""
    dataset = get_benchmark_dataset()

    assert len(dataset) >= 10
    assert all(item.get("question") for item in dataset)
    assert any("áp xe" in item["question"].lower() for item in dataset)
