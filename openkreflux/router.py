"""
OpenKreflux Resilient Multi-Provider Inference Router (Fluko).

Features:
- Provider definitions (endpoints, credentials, models, priority, timeouts).
- Provider health tracking with exponential failure backoff and EWMA latency tracking.
- Automatic failover across providers (e.g. Featherless, Neokens, OpenRouter).
- SSE streaming chunk aggregator with connection drop detection and mid-stream failover resumption.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Union

import httpx

logger = logging.getLogger("openkreflux.router")

CAPACITY_ERROR_CODES = {408, 409, 425, 429, 502, 503, 504}
CAPACITY_ERROR_REGEX = re.compile(
    r"concurrent(?: request)? limit|too many concurrent|rate limit|capacity|overloaded|temporarily unavailable|quota exceeded",
    re.IGNORECASE,
)


class RouterError(Exception):
    """Base error for router operations."""


class AllProvidersFailedError(RouterError):
    """Raised when all configured providers fail or are exhausted."""

    def __init__(self, attempts: List[Dict[str, Any]]):
        self.attempts = attempts
        super().__init__(f"All inference providers failed. Attempts: {attempts}")


class ProviderCapacityError(RouterError):
    """Raised when a provider rejects request due to capacity or rate limits."""


@dataclass
class ProviderConfig:
    """Configuration for an upstream inference provider."""

    name: str
    base_url: str
    api_key: Optional[str] = None
    models: List[str] = field(default_factory=list)
    default_model: Optional[str] = None
    priority: int = 1  # Lower value = higher priority
    timeout: float = 30.0
    max_retries: int = 2
    backoff_factor: float = 1.5
    backoff_max_seconds: float = 60.0
    custom_headers: Dict[str, str] = field(default_factory=dict)

    def get_effective_model(self, requested_model: Optional[str] = None) -> str:
        if requested_model:
            return requested_model
        if self.default_model:
            return self.default_model
        if self.models:
            return self.models[0]
        return "default"


@dataclass
class ProviderStatus:
    """Live health and latency tracking for a provider."""

    consecutive_failures: int = 0
    last_failure_time: Optional[float] = None
    backoff_until: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    ewma_latency_ms: float = 0.0
    last_latency_ms: float = 0.0

    def is_healthy(self, now: Optional[float] = None) -> bool:
        current_time = now if now is not None else time.time()
        return current_time >= self.backoff_until

    def record_success(self, latency_ms: float) -> None:
        self.total_requests += 1
        self.successful_requests += 1
        self.consecutive_failures = 0
        self.backoff_until = 0.0
        self.last_latency_ms = latency_ms
        if self.ewma_latency_ms == 0.0:
            self.ewma_latency_ms = latency_ms
        else:
            # 80% history, 20% newest observation
            self.ewma_latency_ms = 0.8 * self.ewma_latency_ms + 0.2 * latency_ms

    def record_failure(
        self,
        backoff_factor: float = 1.5,
        backoff_max: float = 60.0,
        now: Optional[float] = None,
    ) -> float:
        current_time = now if now is not None else time.time()
        self.total_requests += 1
        self.consecutive_failures += 1
        self.last_failure_time = current_time

        delay = min(backoff_max, (backoff_factor ** self.consecutive_failures))
        self.backoff_until = current_time + delay
        return delay


@dataclass
class StreamChunk:
    """Individual streamed delta token/chunk from a provider."""

    content: str = ""
    reasoning_content: str = ""
    provider: str = ""
    model: str = ""
    is_resumed: bool = False
    is_complete: bool = False
    finish_reason: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class CompletionResponse:
    """Non-streaming aggregated response."""

    content: str
    reasoning_content: Optional[str] = None
    provider: str = ""
    model: str = ""
    latency_ms: float = 0.0
    attempts: List[Dict[str, Any]] = field(default_factory=list)


def is_provider_capacity_failure(status_code: int, response_body: str) -> bool:
    """Determine if HTTP status or body signals capacity/rate limit exhaustion."""
    if status_code in CAPACITY_ERROR_CODES:
        return True
    if response_body and CAPACITY_ERROR_REGEX.search(response_body):
        return True
    return False


def parse_sse_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse standard OpenAI/Fluko SSE data line."""
    clean = line.strip()
    if not clean.startswith("data:"):
        return None
    data_payload = clean[5:].strip()
    if not data_payload or data_payload == "[DONE]":
        return {"done": True}
    try:
        return json.loads(data_payload)
    except json.JSONDecodeError:
        return None


class FlukoRouter:
    """
    Production-grade resilient multi-provider router.

    Automatically prioritizes healthy providers, performs exponential backoff on
    overloaded providers, and manages streaming failover resumption.
    """

    def __init__(
        self,
        providers: Optional[List[ProviderConfig]] = None,
        client: Optional[httpx.Client] = None,
        async_client: Optional[httpx.AsyncClient] = None,
    ):
        self.providers: Dict[str, ProviderConfig] = {}
        self.status: Dict[str, ProviderStatus] = {}
        self._client = client
        self._async_client = async_client

        if providers:
            for p in providers:
                self.register_provider(p)

    def register_provider(self, provider: ProviderConfig) -> None:
        self.providers[provider.name] = provider
        if provider.name not in self.status:
            self.status[provider.name] = ProviderStatus()

    @classmethod
    def create_default(
        cls,
        featherless_key: Optional[str] = None,
        neokens_key: Optional[str] = None,
        openrouter_key: Optional[str] = None,
    ) -> FlukoRouter:
        """Create standard router pre-configured with Featherless, Neokens, and OpenRouter."""
        configs = [
            ProviderConfig(
                name="featherless",
                base_url="https://api.featherless.ai/v1",
                api_key=featherless_key,
                models=["glm-4-9b-chat", "meta-llama/Llama-3.3-70B-Instruct"],
                default_model="glm-4-9b-chat",
                priority=1,
                timeout=20.0,
            ),
            ProviderConfig(
                name="openrouter",
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key,
                models=["anthropic/claude-3.5-sonnet", "deepseek/deepseek-r1"],
                default_model="deepseek/deepseek-r1",
                priority=2,
                timeout=28.0,
            ),
            ProviderConfig(
                name="neokens",
                base_url="https://api.neokens.com/v1",
                api_key=neokens_key,
                models=["gpt-4o", "claude-3-5-sonnet-20241022"],
                default_model="gpt-4o",
                priority=3,
                timeout=22.0,
            ),
        ]
        return cls(providers=configs)

    def get_ordered_providers(
        self, requested_model: Optional[str] = None, now: Optional[float] = None
    ) -> List[ProviderConfig]:
        """
        Rank providers:
        1. Healthy providers first (sorted by priority ascending, then EWMA latency).
        2. Unhealthy/backoff providers next (sorted by backoff_until ascending).
        """
        current_time = now if now is not None else time.time()
        candidates = list(self.providers.values())

        def sort_key(p: ProviderConfig):
            st = self.status[p.name]
            healthy = st.is_healthy(current_time)
            # Tuple: (not_healthy, priority, ewma_latency, backoff_until)
            return (
                0 if healthy else 1,
                p.priority,
                st.ewma_latency_ms,
                st.backoff_until,
            )

        return sorted(candidates, key=sort_key)

    def _get_http_client(self) -> httpx.Client:
        if self._client is not None and not self._client.is_closed:
            return self._client
        return httpx.Client()

    def _get_async_http_client(self) -> httpx.AsyncClient:
        if self._async_client is not None and not self._async_client.is_closed:
            return self._async_client
        return httpx.AsyncClient()

    def _prepare_headers(self, provider: ProviderConfig) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"
        headers.update(provider.custom_headers)
        return headers

    def complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> CompletionResponse:
        """
        Execute synchronous chat completion with automatic provider fallback.
        """
        attempts: List[Dict[str, Any]] = []
        ordered_providers = self.get_ordered_providers(requested_model=model)

        if not ordered_providers:
            raise RouterError("No providers registered in FlukoRouter")

        client = self._get_http_client()
        owns_client = self._client is None

        try:
            for provider in ordered_providers:
                target_model = provider.get_effective_model(model)
                payload: Dict[str, Any] = {
                    "model": target_model,
                    "messages": messages,
                    "temperature": temperature,
                    "stream": False,
                }
                if max_tokens is not None:
                    payload["max_tokens"] = max_tokens
                if extra_body:
                    payload.update(extra_body)

                url = f"{provider.base_url.rstrip('/')}/chat/completions"
                headers = self._prepare_headers(provider)
                start_time = time.time()

                attempt_record: Dict[str, Any] = {
                    "provider": provider.name,
                    "model": target_model,
                    "timestamp": start_time,
                }

                try:
                    resp = client.post(
                        url,
                        json=payload,
                        headers=headers,
                        timeout=provider.timeout,
                    )
                    elapsed_ms = (time.time() - start_time) * 1000.0

                    if resp.status_code == 200:
                        data = resp.json()
                        choice = data.get("choices", [{}])[0]
                        message = choice.get("message", {})
                        content = message.get("content", "")
                        reasoning = message.get("reasoning_content") or message.get("reasoning")

                        self.status[provider.name].record_success(elapsed_ms)
                        attempt_record["status"] = "success"
                        attempt_record["latency_ms"] = elapsed_ms
                        attempts.append(attempt_record)

                        return CompletionResponse(
                            content=content or "",
                            reasoning_content=reasoning,
                            provider=provider.name,
                            model=target_model,
                            latency_ms=elapsed_ms,
                            attempts=attempts,
                        )

                    # Handle error status
                    is_capacity = is_provider_capacity_failure(resp.status_code, resp.text)
                    backoff = self.status[provider.name].record_failure(
                        backoff_factor=provider.backoff_factor,
                        backoff_max=provider.backoff_max_seconds,
                    )
                    attempt_record["status"] = "capacity_error" if is_capacity else f"http_{resp.status_code}"
                    attempt_record["error"] = resp.text[:200]
                    attempt_record["backoff_seconds"] = backoff
                    attempts.append(attempt_record)
                    logger.warning(
                        f"Provider {provider.name} failed with HTTP {resp.status_code}. Backing off {backoff:.1f}s."
                    )

                except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPError) as err:
                    backoff = self.status[provider.name].record_failure(
                        backoff_factor=provider.backoff_factor,
                        backoff_max=provider.backoff_max_seconds,
                    )
                    attempt_record["status"] = "network_error"
                    attempt_record["error"] = str(err)
                    attempt_record["backoff_seconds"] = backoff
                    attempts.append(attempt_record)
                    logger.warning(
                        f"Provider {provider.name} network exception: {err}. Backing off {backoff:.1f}s."
                    )

            raise AllProvidersFailedError(attempts=attempts)
        finally:
            if owns_client:
                client.close()

    async def acomplete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> CompletionResponse:
        """
        Execute asynchronous chat completion with automatic provider fallback.
        """
        attempts: List[Dict[str, Any]] = []
        ordered_providers = self.get_ordered_providers(requested_model=model)

        if not ordered_providers:
            raise RouterError("No providers registered in FlukoRouter")

        client = self._get_async_http_client()
        owns_client = self._async_client is None

        try:
            for provider in ordered_providers:
                target_model = provider.get_effective_model(model)
                payload: Dict[str, Any] = {
                    "model": target_model,
                    "messages": messages,
                    "temperature": temperature,
                    "stream": False,
                }
                if max_tokens is not None:
                    payload["max_tokens"] = max_tokens
                if extra_body:
                    payload.update(extra_body)

                url = f"{provider.base_url.rstrip('/')}/chat/completions"
                headers = self._prepare_headers(provider)
                start_time = time.time()

                attempt_record: Dict[str, Any] = {
                    "provider": provider.name,
                    "model": target_model,
                    "timestamp": start_time,
                }

                try:
                    resp = await client.post(
                        url,
                        json=payload,
                        headers=headers,
                        timeout=provider.timeout,
                    )
                    elapsed_ms = (time.time() - start_time) * 1000.0

                    if resp.status_code == 200:
                        data = resp.json()
                        choice = data.get("choices", [{}])[0]
                        message = choice.get("message", {})
                        content = message.get("content", "")
                        reasoning = message.get("reasoning_content") or message.get("reasoning")

                        self.status[provider.name].record_success(elapsed_ms)
                        attempt_record["status"] = "success"
                        attempt_record["latency_ms"] = elapsed_ms
                        attempts.append(attempt_record)

                        return CompletionResponse(
                            content=content or "",
                            reasoning_content=reasoning,
                            provider=provider.name,
                            model=target_model,
                            latency_ms=elapsed_ms,
                            attempts=attempts,
                        )

                    is_capacity = is_provider_capacity_failure(resp.status_code, resp.text)
                    backoff = self.status[provider.name].record_failure(
                        backoff_factor=provider.backoff_factor,
                        backoff_max=provider.backoff_max_seconds,
                    )
                    attempt_record["status"] = "capacity_error" if is_capacity else f"http_{resp.status_code}"
                    attempt_record["error"] = resp.text[:200]
                    attempt_record["backoff_seconds"] = backoff
                    attempts.append(attempt_record)

                except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPError) as err:
                    backoff = self.status[provider.name].record_failure(
                        backoff_factor=provider.backoff_factor,
                        backoff_max=provider.backoff_max_seconds,
                    )
                    attempt_record["status"] = "network_error"
                    attempt_record["error"] = str(err)
                    attempt_record["backoff_seconds"] = backoff
                    attempts.append(attempt_record)

            raise AllProvidersFailedError(attempts=attempts)
        finally:
            if owns_client:
                await client.aclose()

    def stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        failover_resumption: bool = True,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Iterator[StreamChunk]:
        """
        Stream completion chunks with dropped-connection detection and failover resumption.
        """
        ordered_providers = self.get_ordered_providers(requested_model=model)
        if not ordered_providers:
            raise RouterError("No providers registered in FlukoRouter")

        client = self._get_http_client()
        owns_client = self._client is None

        accumulated_content = ""
        accumulated_reasoning = ""
        provider_index = 0
        attempts: List[Dict[str, Any]] = []

        try:
            while provider_index < len(ordered_providers):
                provider = ordered_providers[provider_index]
                target_model = provider.get_effective_model(model)

                # Prepare messages (if resuming after dropout, include partial response context)
                active_messages = list(messages)
                is_resumed_attempt = bool(accumulated_content or accumulated_reasoning)
                if is_resumed_attempt:
                    # Instruct continuity from where generation dropped off
                    prefix_instruction = ""
                    if accumulated_reasoning:
                        prefix_instruction += f"<think>\n{accumulated_reasoning}\n</think>\n"
                    if accumulated_content:
                        prefix_instruction += accumulated_content
                    active_messages.append({
                        "role": "assistant",
                        "content": prefix_instruction,
                    })
                    active_messages.append({
                        "role": "user",
                        "content": "[Connection interrupted. Please continue smoothly from the exact last word without repeating previous output.]",
                    })

                payload: Dict[str, Any] = {
                    "model": target_model,
                    "messages": active_messages,
                    "temperature": temperature,
                    "stream": True,
                }
                if max_tokens is not None:
                    payload["max_tokens"] = max_tokens
                if extra_body:
                    payload.update(extra_body)

                url = f"{provider.base_url.rstrip('/')}/chat/completions"
                headers = self._prepare_headers(provider)
                headers["Accept"] = "text/event-stream"
                start_time = time.time()
                got_first_token = False

                attempt_record: Dict[str, Any] = {
                    "provider": provider.name,
                    "model": target_model,
                    "is_resumption": is_resumed_attempt,
                }

                try:
                    with client.stream(
                        "POST",
                        url,
                        json=payload,
                        headers=headers,
                        timeout=provider.timeout,
                    ) as response:
                        if response.status_code != 200:
                            body = response.read().decode(errors="ignore")
                            is_capacity = is_provider_capacity_failure(response.status_code, body)
                            self.status[provider.name].record_failure(
                                backoff_factor=provider.backoff_factor,
                                backoff_max=provider.backoff_max_seconds,
                            )
                            attempt_record["status"] = "capacity_error" if is_capacity else f"http_{response.status_code}"
                            attempts.append(attempt_record)
                            provider_index += 1
                            continue

                        for line in response.iter_lines():
                            parsed = parse_sse_line(line)
                            if parsed is None:
                                continue
                            if parsed.get("done"):
                                elapsed = (time.time() - start_time) * 1000.0
                                self.status[provider.name].record_success(elapsed)
                                yield StreamChunk(
                                    content="",
                                    provider=provider.name,
                                    model=target_model,
                                    is_resumed=is_resumed_attempt,
                                    is_complete=True,
                                    finish_reason="stop",
                                    latency_ms=elapsed,
                                )
                                return

                            choices = parsed.get("choices", [])
                            if not choices:
                                continue
                            choice = choices[0]
                            delta = choice.get("delta", {})
                            content_piece = delta.get("content", "") or ""
                            reasoning_piece = (
                                delta.get("reasoning_content")
                                or delta.get("reasoning")
                                or ""
                            )
                            finish_reason = choice.get("finish_reason")

                            if content_piece or reasoning_piece:
                                got_first_token = True
                                accumulated_content += content_piece
                                accumulated_reasoning += reasoning_piece

                                yield StreamChunk(
                                    content=content_piece,
                                    reasoning_content=reasoning_piece,
                                    provider=provider.name,
                                    model=target_model,
                                    is_resumed=is_resumed_attempt,
                                    is_complete=bool(finish_reason),
                                    finish_reason=finish_reason,
                                    latency_ms=(time.time() - start_time) * 1000.0,
                                )

                            if finish_reason:
                                elapsed = (time.time() - start_time) * 1000.0
                                self.status[provider.name].record_success(elapsed)
                                return

                except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPError, httpx.StreamError) as err:
                    # Connection dropped mid-stream or before stream started
                    self.status[provider.name].record_failure(
                        backoff_factor=provider.backoff_factor,
                        backoff_max=provider.backoff_max_seconds,
                    )
                    attempt_record["status"] = "stream_dropped" if got_first_token else "connection_failed"
                    attempt_record["error"] = str(err)
                    attempts.append(attempt_record)
                    logger.warning(
                        f"Stream error on {provider.name} (got_first_token={got_first_token}): {err}"
                    )

                    if not failover_resumption:
                        raise

                    provider_index += 1
                    continue

                # Finished normally
                return

            raise AllProvidersFailedError(attempts=attempts)
        finally:
            if owns_client:
                client.close()

    async def astream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        failover_resumption: bool = True,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Async streaming with dropped-connection detection and failover resumption.
        """
        ordered_providers = self.get_ordered_providers(requested_model=model)
        if not ordered_providers:
            raise RouterError("No providers registered in FlukoRouter")

        client = self._get_async_http_client()
        owns_client = self._async_client is None

        accumulated_content = ""
        accumulated_reasoning = ""
        provider_index = 0
        attempts: List[Dict[str, Any]] = []

        try:
            while provider_index < len(ordered_providers):
                provider = ordered_providers[provider_index]
                target_model = provider.get_effective_model(model)

                active_messages = list(messages)
                is_resumed_attempt = bool(accumulated_content or accumulated_reasoning)
                if is_resumed_attempt:
                    prefix_instruction = ""
                    if accumulated_reasoning:
                        prefix_instruction += f"<think>\n{accumulated_reasoning}\n</think>\n"
                    if accumulated_content:
                        prefix_instruction += accumulated_content
                    active_messages.append({
                        "role": "assistant",
                        "content": prefix_instruction,
                    })
                    active_messages.append({
                        "role": "user",
                        "content": "[Connection interrupted. Please continue smoothly from the exact last word without repeating previous output.]",
                    })

                payload: Dict[str, Any] = {
                    "model": target_model,
                    "messages": active_messages,
                    "temperature": temperature,
                    "stream": True,
                }
                if max_tokens is not None:
                    payload["max_tokens"] = max_tokens
                if extra_body:
                    payload.update(extra_body)

                url = f"{provider.base_url.rstrip('/')}/chat/completions"
                headers = self._prepare_headers(provider)
                headers["Accept"] = "text/event-stream"
                start_time = time.time()
                got_first_token = False

                attempt_record: Dict[str, Any] = {
                    "provider": provider.name,
                    "model": target_model,
                    "is_resumption": is_resumed_attempt,
                }

                try:
                    async with client.stream(
                        "POST",
                        url,
                        json=payload,
                        headers=headers,
                        timeout=provider.timeout,
                    ) as response:
                        if response.status_code != 200:
                            body = (await response.aread()).decode(errors="ignore")
                            is_capacity = is_provider_capacity_failure(response.status_code, body)
                            self.status[provider.name].record_failure(
                                backoff_factor=provider.backoff_factor,
                                backoff_max=provider.backoff_max_seconds,
                            )
                            attempt_record["status"] = "capacity_error" if is_capacity else f"http_{response.status_code}"
                            attempts.append(attempt_record)
                            provider_index += 1
                            continue

                        async for line in response.aiter_lines():
                            parsed = parse_sse_line(line)
                            if parsed is None:
                                continue
                            if parsed.get("done"):
                                elapsed = (time.time() - start_time) * 1000.0
                                self.status[provider.name].record_success(elapsed)
                                yield StreamChunk(
                                    content="",
                                    provider=provider.name,
                                    model=target_model,
                                    is_resumed=is_resumed_attempt,
                                    is_complete=True,
                                    finish_reason="stop",
                                    latency_ms=elapsed,
                                )
                                return

                            choices = parsed.get("choices", [])
                            if not choices:
                                continue
                            choice = choices[0]
                            delta = choice.get("delta", {})
                            content_piece = delta.get("content", "") or ""
                            reasoning_piece = (
                                delta.get("reasoning_content")
                                or delta.get("reasoning")
                                or ""
                            )
                            finish_reason = choice.get("finish_reason")

                            if content_piece or reasoning_piece:
                                got_first_token = True
                                accumulated_content += content_piece
                                accumulated_reasoning += reasoning_piece

                                yield StreamChunk(
                                    content=content_piece,
                                    reasoning_content=reasoning_piece,
                                    provider=provider.name,
                                    model=target_model,
                                    is_resumed=is_resumed_attempt,
                                    is_complete=bool(finish_reason),
                                    finish_reason=finish_reason,
                                    latency_ms=(time.time() - start_time) * 1000.0,
                                )

                            if finish_reason:
                                elapsed = (time.time() - start_time) * 1000.0
                                self.status[provider.name].record_success(elapsed)
                                return

                except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPError, httpx.StreamError) as err:
                    self.status[provider.name].record_failure(
                        backoff_factor=provider.backoff_factor,
                        backoff_max=provider.backoff_max_seconds,
                    )
                    attempt_record["status"] = "stream_dropped" if got_first_token else "connection_failed"
                    attempt_record["error"] = str(err)
                    attempts.append(attempt_record)

                    if not failover_resumption:
                        raise

                    provider_index += 1
                    continue

                return

            raise AllProvidersFailedError(attempts=attempts)
        finally:
            if owns_client:
                await client.aclose()
