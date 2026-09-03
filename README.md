# OpenKreflux

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](https://github.com/Kreflux/openkreflux)

**OpenKreflux** is the core open-source Python research & inference toolkit powering **[Kreflux](https://kreflux.ai)**.

It provides a production-grade, fault-tolerant inference engine featuring **Fluko's Resilient Multi-Provider Router** (Featherless, Neokens, OpenRouter) with dropped-stream failover resumption, an automated **Reasoning Trace Verifier**, **Reasoning Ladder Scoring** (Low → Ultra), and an **Inference Benchmarking Suite**.

---

## Architecture Overview

```mermaid
flowchart TD
    subgraph Client ["Client Application / CLI"]
        UserReq["User Prompt / Context"]
    end

    subgraph OpenKrefluxEngine ["OpenKreflux Engine"]
        Router["FlukoRouter<br/>(EWMA Latency + Priority)"]
        Health["ProviderStatus<br/>(Exponential Backoff)"]
        StreamAgg["Streaming Aggregator<br/>(Dropout Detector & Resumption)"]
        Verifier["ReasoningVerifier<br/>(<think> Parser & AST / Math Validator)"]
        Ladder["ReasoningLadder<br/>(Low | Medium | High | Ultra)"]
    end

    subgraph Providers ["Upstream Inference Providers"]
        P1["Featherless AI<br/>(Primary Low-Latency)"]
        P2["OpenRouter<br/>(Frontier Failover)"]
        P3["Neokens<br/>(High-Headroom Gateway)"]
    end

    UserReq --> Router
    Router <--> Health
    Router --> P1
    P1 -.->|Capacity / Dropout (429/503)| Router
    Router --> P2
    P2 -.->|Failover| Router
    Router --> P3
    
    P1 & P2 & P3 --> StreamAgg
    StreamAgg --> Verifier
    Verifier --> Ladder
```

---

## Key Features

1. **Fluko Resilient Multi-Provider Routing**:
   - Priority-based and latency-weighted (EWMA) dispatch.
   - Dynamic capacity error classification (`429`, `503`, concurrency limit bodies).
   - Adaptive exponential backoff preventing cascading provider outages.
2. **Streaming Chunk Aggregator with Mid-Stream Resumption**:
   - Detects connection drops mid-generation.
   - Preserves already-streamed tokens and seamlessly hands off to secondary providers to complete generation without restart penalties.
3. **Reasoning Trace Verifier**:
   - Parses native `<think>...</think>` tokens.
   - Checks logical coherence and flags degenerate repetitive n-gram loops.
   - Mathematically validates LaTeX delimiters (`$...$`, `$$...$$`) and bracket balances.
   - Inspects Python code blocks via AST compilation for syntax integrity.
4. **Reasoning Ladder Depth Standard**:
   - Formalizes test-time compute into four standardized rungs:
     - **Low** (Fast quick verification, ~10+ tokens)
     - **Medium** (Balanced multi-step reasoning, ~150+ tokens)
     - **High** (Deep reasoning with self-correction, ~500+ tokens)
     - **Ultra** (Rigorous proofs, branch exploration, edge-case audit, ~1200+ tokens)
   - Computes **Reasoning Density Score** (`thinking_tokens / total_tokens`).
5. **Inference Benchmarking Suite**:
   - Accurate measurement of **Time To First Token (TTFT)**, **Tokens Per Second (TPS)**, and **Failover Latency Overhead**.
6. **Rich Interactive CLI**:
   - CLI commands for inspecting architecture, verifying trace files, and benchmarking backends.

---

## Installation

### Using pip
```bash
pip install openkreflux
```

### Using uv (recommended)
```bash
uv pip install openkreflux
```

### Development setup
```bash
git clone https://github.com/Kreflux/openkreflux.git
cd openkreflux
uv pip install -e ".[dev]"
```

---

## Quickstart Guide

### 1. Resilient Multi-Provider Completion

```python
from openkreflux import FlukoRouter, ProviderConfig

# Initialize router with fallback providers
router = FlukoRouter([
    ProviderConfig(
        name="featherless",
        base_url="https://api.featherless.ai/v1",
        api_key="YOUR_FEATHERLESS_KEY",
        priority=1,
        timeout=20.0,
    ),
    ProviderConfig(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key="YOUR_OPENROUTER_KEY",
        priority=2,
        timeout=28.0,
    ),
])

# Complete with automatic failover
response = router.complete(
    messages=[{"role": "user", "content": "Explain quantum entanglement in 2 sentences."}],
    model="kreflux-preview",
)

print(f"Provider used: {response.provider}")
print(f"Latency: {response.latency_ms:.1f}ms")
print(f"Response:\n{response.content}")
```

### 2. Streaming with Drop Resumption

```python
for chunk in router.stream(
    messages=[{"role": "user", "content": "Derive Euler's formula step by step."}],
    failover_resumption=True,
):
    if chunk.reasoning_content:
        print(f"[Thinking: {chunk.reasoning_content}]", end="", flush=True)
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

### 3. Reasoning Trace Verification

```python
from openkreflux import ReasoningVerifier, LadderLevel

verifier = ReasoningVerifier()

trace = """
<think>
Let us solve 2x + 6 = 14.
Step 1: Subtract 6 from both sides: 2x = 8.
Step 2: Divide both sides by 2: x = 4.
Let me double check by substitution: 2(4) + 6 = 8 + 6 = 14.
Matches original equation.
</think>
The solution is $x = 4$.
"""

result = verifier.verify(trace, target_ladder=LadderLevel.MEDIUM)

print(f"Is Valid: {result.is_valid}")
print(f"Ladder Level: {result.ladder_level.value}")
print(f"Reasoning Density: {result.reasoning_density * 100:.1f}%")
print(f"Thinking Tokens: {result.thinking_tokens}")
print(f"Self-Correction: {result.has_self_correction}")
```

---

## CLI Usage

### 1. View System & Architecture Info
```bash
openkreflux info
```

### 2. Verify Reasoning Trace File
```bash
openkreflux verify-trace trace.txt --target-level high --show-thoughts
```

Example output:
```text
✔ VALID REASONING TRACE (Ladder: HIGH)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric                   ┃ Value                     ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Thinking Tokens          │ 612                       │
│ Solution Tokens          │ 145                       │
│ Total Tokens             │ 757                       │
│ Reasoning Density        │ 80.8% (████████████████░░░) │
│ Coherence Score          │ 0.85 / 1.00               │
│ Self-Correction Detected │ Yes                       │
│ Math Syntax & Delimiters │ Valid                     │
│ Code AST Syntax          │ Valid                     │
│ Meets Target ('high')    │ Satisfied                 │
└──────────────────────────┴───────────────────────────┘
```

### 3. Benchmark Inference Engine
```bash
# Synthetic resilience benchmark (no API keys required)
openkreflux benchmark --model kreflux-preview --provider mock

# Real provider benchmark
export FEATHERLESS_API_KEY="your-key"
openkreflux benchmark --model kreflux-preview --provider featherless --iterations 3
```

---

## Running Tests

Run the test suite using `pytest`:

```bash
pytest
```

Or using `uv`:
```bash
uv run pytest -v
```

---

## Contributing

Contributions are welcome! Please open an issue or pull request at [Kreflux/openkreflux](https://github.com/Kreflux/openkreflux).

## License

OpenKreflux is licensed under the [Apache 2.0 License](LICENSE).
