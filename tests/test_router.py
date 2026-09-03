"""
Unit tests for FlukoRouter: failover, health tracking, and stream resumption.
"""

import json
import time
import pytest
import httpx

from openkreflux.router import (
    AllProvidersFailedError,
    FlukoRouter,
    ProviderConfig,
    ProviderStatus,
    StreamChunk,
    is_provider_capacity_failure,
    parse_sse_line,
)


def test_capacity_failure_detection():
    # HTTP status code detection
    assert is_provider_capacity_failure(429, "") is True
    assert is_provider_capacity_failure(503, "") is True
    assert is_provider_capacity_failure(502, "") is True
    assert is_provider_capacity_failure(408, "") is True
    assert is_provider_capacity_failure(200, "") is False
    assert is_provider_capacity_failure(400, "Bad Request") is False

    # Body string regex detection
    assert is_provider_capacity_failure(400, "Concurrent request limit exceeded") is True
    assert is_provider_capacity_failure(200, "rate limit reached, please slow down") is True
    assert is_provider_capacity_failure(500, "server capacity temporarily unavailable") is True


def test_provider_status_health_and_backoff():
    status = ProviderStatus()
    assert status.is_healthy(now=100.0) is True

    # Record first failure
    delay1 = status.record_failure(backoff_factor=2.0, backoff_max=30.0, now=100.0)
    assert delay1 == 2.0
    assert status.backoff_until == 102.0
    assert status.is_healthy(now=101.0) is False
    assert status.is_healthy(now=102.5) is True

    # Record second failure
    delay2 = status.record_failure(backoff_factor=2.0, backoff_max=30.0, now=102.5)
    assert delay2 == 4.0
    assert status.backoff_until == 106.5
    assert status.is_healthy(now=105.0) is False
    assert status.is_healthy(now=107.0) is True

    # Record success resets backoff
    status.record_success(latency_ms=150.0)
    assert status.consecutive_failures == 0
    assert status.backoff_until == 0.0
    assert status.is_healthy(now=107.0) is True
    assert status.ewma_latency_ms == 150.0

    # Second success updates EWMA (80% / 20%)
    status.record_success(latency_ms=100.0)
    assert pytest.approx(status.ewma_latency_ms, rel=1e-2) == 140.0  # 0.8 * 150 + 0.2 * 100


def test_provider_ordering_priority_and_health():
    router = FlukoRouter()
    p1 = ProviderConfig(name="primary", base_url="http://p1", priority=1)
    p2 = ProviderConfig(name="secondary", base_url="http://p2", priority=2)
    p3 = ProviderConfig(name="tertiary", base_url="http://p3", priority=3)

    router.register_provider(p3)
    router.register_provider(p1)
    router.register_provider(p2)

    # Initial order should be p1, p2, p3 based on priority
    ordered = router.get_ordered_providers(now=100.0)
    assert [p.name for p in ordered] == ["primary", "secondary", "tertiary"]

    # Mark p1 as failed with backoff until 200.0
    router.status["primary"].backoff_until = 200.0

    # At t=150.0, p1 is unhealthy, so p2 and p3 take precedence
    ordered = router.get_ordered_providers(now=150.0)
    assert [p.name for p in ordered] == ["secondary", "tertiary", "primary"]

    # At t=250.0, p1's backoff expired, so it regains primary spot
    ordered = router.get_ordered_providers(now=250.0)
    assert [p.name for p in ordered] == ["primary", "secondary", "tertiary"]


def test_sse_line_parsing():
    assert parse_sse_line("") is None
    assert parse_sse_line("event: ping") is None
    assert parse_sse_line("data: [DONE]") == {"done": True}

    payload = {"choices": [{"delta": {"content": "hello"}}]}
    line = f"data: {json.dumps(payload)}"
    parsed = parse_sse_line(line)
    assert parsed == payload


def test_router_complete_fallback():
    # Mock transport simulating primary failing with 429 and secondary succeeding
    call_counts = {"primary": 0, "secondary": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "primary" in url_str:
            call_counts["primary"] += 1
            return httpx.Response(
                429,
                text="Rate limit exceeded: concurrent requests cap",
            )
        elif "secondary" in url_str:
            call_counts["secondary"] += 1
            body = {
                "choices": [
                    {
                        "message": {
                            "content": "Kreflux solved this.",
                            "reasoning_content": "Detailed reasoning steps.",
                        }
                    }
                ]
            }
            return httpx.Response(200, json=body)
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    router = FlukoRouter(client=client)

    p1 = ProviderConfig(name="primary", base_url="http://primary/v1", priority=1)
    p2 = ProviderConfig(name="secondary", base_url="http://secondary/v1", priority=2)
    router.register_provider(p1)
    router.register_provider(p2)

    resp = router.complete(messages=[{"role": "user", "content": "hello"}])

    assert resp.provider == "secondary"
    assert resp.content == "Kreflux solved this."
    assert resp.reasoning_content == "Detailed reasoning steps."
    assert call_counts["primary"] == 1
    assert call_counts["secondary"] == 1
    assert len(resp.attempts) == 2
    assert resp.attempts[0]["status"] == "capacity_error"
    assert resp.attempts[1]["status"] == "success"


def test_router_all_providers_failed():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="Service Unavailable")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    router = FlukoRouter(client=client)
    router.register_provider(ProviderConfig(name="p1", base_url="http://p1/v1", priority=1))

    with pytest.raises(AllProvidersFailedError) as exc_info:
        router.complete(messages=[{"role": "user", "content": "hi"}])
    assert len(exc_info.value.attempts) == 1


def test_router_streaming_and_failover_resumption():
    # Simulate primary streaming 2 tokens then dropping connection;
    # secondary then picks up and completes the rest.
    p1_yielded = 0

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "primary" in url_str:
            # First chunk succeeds, then connection disconnects
            content = (
                b"data: "
                + json.dumps({"choices": [{"delta": {"content": "Part 1 "}}]}).encode()
                + b"\n\n"
            )
            # Return response that errors on next read by closing
            def stream_gen():
                yield content
                raise httpx.StreamError("Connection dropped by remote peer")

            return httpx.Response(200, content=stream_gen())
        elif "secondary" in url_str:
            # Secondary streams remainder
            content = (
                b"data: "
                + json.dumps({"choices": [{"delta": {"content": "Part 2"}}]}).encode()
                + b"\n\n"
                + b"data: [DONE]\n\n"
            )
            return httpx.Response(200, content=content)
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    router = FlukoRouter(client=client)

    p1 = ProviderConfig(name="primary", base_url="http://primary/v1", priority=1)
    p2 = ProviderConfig(name="secondary", base_url="http://secondary/v1", priority=2)
    router.register_provider(p1)
    router.register_provider(p2)

    chunks = list(
        router.stream(
            messages=[{"role": "user", "content": "start"}],
            failover_resumption=True,
        )
    )

    contents = [c.content for c in chunks if c.content]
    assert contents == ["Part 1 ", "Part 2"]

    # Verify that resumption chunk was identified
    resumed_chunks = [c for c in chunks if c.is_resumed]
    assert len(resumed_chunks) > 0
    assert resumed_chunks[0].provider == "secondary"
