"""
OpenKreflux Inference Benchmarking Suite.

Measures:
- TTFT (Time To First Token).
- Generation speed in Tokens Per Second (TPS).
- Reasoning token ratio (reasoning tokens vs total tokens).
- Failover latency and recovery duration.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from openkreflux.router import FlukoRouter, ProviderConfig, StreamChunk


class BenchmarkMetric(BaseModel):
    """Metrics recorded from a single inference request."""

    ttft_ms: float = Field(description="Time to first token in milliseconds")
    tps: float = Field(description="Generation tokens per second")
    total_time_ms: float = Field(description="Total request turnaround in milliseconds")
    total_tokens: int = Field(description="Total tokens received")
    reasoning_tokens: int = Field(description="Reasoning tokens received")
    solution_tokens: int = Field(description="Solution/content tokens received")
    reasoning_token_ratio: float = Field(description="Ratio of reasoning tokens to total tokens")
    failover_occurred: bool = Field(default=False, description="Whether failover happened")
    failover_latency_ms: float = Field(
        default=0.0, description="Latency overhead spent during failover"
    )
    provider_used: str = Field(default="", description="Provider that satisfied request")
    model_used: str = Field(default="", description="Model served")


class BenchmarkReport(BaseModel):
    """Aggregated statistics across multiple benchmark iterations."""

    iterations: int
    mean_ttft_ms: float
    median_ttft_ms: float
    p95_ttft_ms: float
    mean_tps: float
    mean_reasoning_ratio: float
    mean_total_time_ms: float
    failover_count: int
    mean_failover_latency_ms: float
    runs: List[BenchmarkMetric] = Field(default_factory=list)


DEFAULT_BENCHMARK_PROMPTS = [
    "Explain the difference between optimistic concurrency control and pessimistic locking.",
    "Solve for x: 3x^2 - 12x + 9 = 0. Show your step-by-step reasoning.",
    "Write a Python function to compute the longest palindromic substring with O(n) space.",
]


class InferenceBenchmark:
    """Benchmark suite runner for testing inference latency and provider resilience."""

    def __init__(self, router: FlukoRouter):
        self.router = router

    def run_stream_benchmark(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        failover_resumption: bool = True,
    ) -> BenchmarkMetric:
        """
        Execute streaming inference and collect fine-grained timing metrics.
        """
        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        end_time: Optional[float] = None

        total_reasoning_tokens = 0
        total_solution_tokens = 0
        provider_used = ""
        model_used = ""
        failover_occurred = False
        failover_latency_ms = 0.0

        last_chunk_time = start_time

        chunks = self.router.stream(
            messages=messages,
            model=model,
            failover_resumption=failover_resumption,
        )

        for chunk in chunks:
            now = time.perf_counter()
            if first_token_time is None and (chunk.content or chunk.reasoning_content):
                first_token_time = now

            if chunk.is_resumed and not failover_occurred:
                failover_occurred = True
                failover_latency_ms = (now - last_chunk_time) * 1000.0

            if chunk.reasoning_content:
                # Count approximate tokens in chunk
                tokens_count = max(1, len(chunk.reasoning_content.split()))
                total_reasoning_tokens += tokens_count

            if chunk.content:
                tokens_count = max(1, len(chunk.content.split()))
                total_solution_tokens += tokens_count

            if chunk.provider:
                provider_used = chunk.provider
            if chunk.model:
                model_used = chunk.model

            last_chunk_time = now

        end_time = time.perf_counter()

        ttft_ms = (
            (first_token_time - start_time) * 1000.0
            if first_token_time is not None
            else (end_time - start_time) * 1000.0
        )
        total_time_ms = (end_time - start_time) * 1000.0
        total_tokens = total_reasoning_tokens + total_solution_tokens

        generation_time_sec = (
            (end_time - first_token_time)
            if first_token_time is not None
            else (end_time - start_time)
        )
        tps = (
            (total_tokens / generation_time_sec)
            if generation_time_sec > 0 and total_tokens > 0
            else 0.0
        )

        ratio = (
            (total_reasoning_tokens / total_tokens) if total_tokens > 0 else 0.0
        )

        return BenchmarkMetric(
            ttft_ms=round(ttft_ms, 2),
            tps=round(tps, 2),
            total_time_ms=round(total_time_ms, 2),
            total_tokens=total_tokens,
            reasoning_tokens=total_reasoning_tokens,
            solution_tokens=total_solution_tokens,
            reasoning_token_ratio=round(ratio, 4),
            failover_occurred=failover_occurred,
            failover_latency_ms=round(failover_latency_ms, 2),
            provider_used=provider_used,
            model_used=model_used,
        )

    def run_suite(
        self,
        prompts: Optional[List[str]] = None,
        model: Optional[str] = None,
        iterations_per_prompt: int = 2,
    ) -> BenchmarkReport:
        """
        Execute benchmark over a suite of prompts and compute statistics.
        """
        test_prompts = prompts or DEFAULT_BENCHMARK_PROMPTS
        results: List[BenchmarkMetric] = []

        for p in test_prompts:
            messages = [{"role": "user", "content": p}]
            for _ in range(iterations_per_prompt):
                metric = self.run_stream_benchmark(messages=messages, model=model)
                results.append(metric)

        if not results:
            return BenchmarkReport(
                iterations=0,
                mean_ttft_ms=0.0,
                median_ttft_ms=0.0,
                p95_ttft_ms=0.0,
                mean_tps=0.0,
                mean_reasoning_ratio=0.0,
                mean_total_time_ms=0.0,
                failover_count=0,
                mean_failover_latency_ms=0.0,
                runs=[],
            )

        ttft_values = [r.ttft_ms for r in results]
        tps_values = [r.tps for r in results]
        ratio_values = [r.reasoning_token_ratio for r in results]
        total_time_values = [r.total_time_ms for r in results]
        failover_runs = [r for r in results if r.failover_occurred]

        p95_index = max(0, int(len(ttft_values) * 0.95) - 1)
        sorted_ttft = sorted(ttft_values)
        p95_ttft = sorted_ttft[p95_index] if sorted_ttft else 0.0

        return BenchmarkReport(
            iterations=len(results),
            mean_ttft_ms=round(statistics.mean(ttft_values), 2),
            median_ttft_ms=round(statistics.median(ttft_values), 2),
            p95_ttft_ms=round(p95_ttft, 2),
            mean_tps=round(statistics.mean(tps_values), 2),
            mean_reasoning_ratio=round(statistics.mean(ratio_values), 4),
            mean_total_time_ms=round(statistics.mean(total_time_values), 2),
            failover_count=len(failover_runs),
            mean_failover_latency_ms=round(
                statistics.mean([r.failover_latency_ms for r in failover_runs])
                if failover_runs
                else 0.0,
                2,
            ),
            runs=results,
        )
