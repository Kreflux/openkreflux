"""
OpenKreflux: Flagship Open-Source Python Research & Inference Toolkit for Kreflux.

Provides:
- KrefluxRouter: Resilient multi-provider routing with automatic failover and stream resumption.
- ReasoningVerifier: Trace verification for <think> tags, math/code validity, and coherence.
- ReasoningLadder: Depth standards and evaluation across Low, Medium, High, and Ultra.
- InferenceBenchmark: Metrics suite measuring TTFT, TPS, reasoning density, and failover latency.
"""

from __future__ import annotations

from openkreflux.benchmark import BenchmarkMetric, BenchmarkReport, InferenceBenchmark
from openkreflux.router import (
    AllProvidersFailedError,
    CompletionResponse,
    KrefluxRouter,
    ProviderCapacityError,
    ProviderConfig,
    ProviderStatus,
    RouterError,
    StreamChunk,
)
from openkreflux.verifier import (
    LadderLevel,
    ReasoningLadder,
    ReasoningVerifier,
    VerificationResult,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "KrefluxRouter",
    "ProviderConfig",
    "ProviderStatus",
    "CompletionResponse",
    "StreamChunk",
    "RouterError",
    "AllProvidersFailedError",
    "ProviderCapacityError",
    "ReasoningVerifier",
    "VerificationResult",
    "ReasoningLadder",
    "LadderLevel",
    "InferenceBenchmark",
    "BenchmarkMetric",
    "BenchmarkReport",
]
