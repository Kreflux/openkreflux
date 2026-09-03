"""
Unit tests for InferenceBenchmark suite and metric collection.
"""

import time
import pytest
from typing import Iterator

from openkreflux.benchmark import BenchmarkMetric, BenchmarkReport, InferenceBenchmark
from openkreflux.router import FlukoRouter, ProviderConfig, StreamChunk


class MockStreamRouter(FlukoRouter):
    """Synthetic router that yields controlled chunks for benchmark testing."""

    def __init__(self, with_failover: bool = False):
        super().__init__()
        self.with_failover = with_failover

    def stream(
        self,
        messages,
        model=None,
        temperature=0.7,
        max_tokens=None,
        failover_resumption=True,
        extra_body=None,
    ) -> Iterator[StreamChunk]:
        # Emulate initial latency for TTFT
        time.sleep(0.02)
        yield StreamChunk(
            reasoning_content="Analyzing step 1",
            provider="mock-provider-1",
            model="kreflux-preview",
            latency_ms=20.0,
        )

        time.sleep(0.01)
        if self.with_failover:
            # Emulate failover resumption chunk
            yield StreamChunk(
                content="Resumed solution part",
                provider="mock-provider-2",
                model="kreflux-preview",
                is_resumed=True,
                latency_ms=30.0,
            )
        else:
            yield StreamChunk(
                content="Solution completed directly",
                provider="mock-provider-1",
                model="kreflux-preview",
                is_resumed=False,
                latency_ms=30.0,
            )

        yield StreamChunk(
            content="",
            provider="mock-provider-1" if not self.with_failover else "mock-provider-2",
            is_complete=True,
            finish_reason="stop",
        )


def test_stream_benchmark_metrics():
    router = MockStreamRouter(with_failover=False)
    bench = InferenceBenchmark(router)

    metric = bench.run_stream_benchmark(
        messages=[{"role": "user", "content": "test prompt"}]
    )

    assert metric.ttft_ms > 0.0
    assert metric.total_tokens > 0
    assert metric.reasoning_tokens > 0
    assert metric.solution_tokens > 0
    assert 0.0 < metric.reasoning_token_ratio < 1.0
    assert metric.tps > 0.0
    assert metric.failover_occurred is False
    assert metric.failover_latency_ms == 0.0


def test_stream_benchmark_failover_detection():
    router = MockStreamRouter(with_failover=True)
    bench = InferenceBenchmark(router)

    metric = bench.run_stream_benchmark(
        messages=[{"role": "user", "content": "test prompt"}]
    )

    assert metric.failover_occurred is True
    assert metric.failover_latency_ms > 0.0
    assert metric.provider_used == "mock-provider-2"


def test_benchmark_suite_aggregation():
    router = MockStreamRouter(with_failover=False)
    bench = InferenceBenchmark(router)

    report = bench.run_suite(
        prompts=["Prompt 1", "Prompt 2"],
        iterations_per_prompt=2,
    )

    assert report.iterations == 4
    assert len(report.runs) == 4
    assert report.mean_ttft_ms > 0.0
    assert report.median_ttft_ms > 0.0
    assert report.p95_ttft_ms > 0.0
    assert report.mean_tps > 0.0
    assert report.failover_count == 0
