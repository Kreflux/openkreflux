"""
OpenKreflux CLI - Rich & Click interactive terminal for OpenKreflux.

Commands:
- openkreflux info
- openkreflux verify-trace <file>
- openkreflux benchmark --model ... --provider ...
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import click
import httpx
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from openkreflux import __version__
from openkreflux.benchmark import InferenceBenchmark
from openkreflux.router import FlukoRouter, ProviderConfig
from openkreflux.verifier import LadderLevel, ReasoningVerifier

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="openkreflux")
def main():
    """OpenKreflux: Resilient multi-provider inference & reasoning verifier toolkit."""


@main.command(name="info")
def info():
    """Display OpenKreflux framework status, providers, and reasoning ladders."""
    banner = Text(
        r"""
  ___                    _  __           __ _             
 / _ \ _ __   ___ _ __  | |/ /_ __ ___  / _| |_   ___  __
| | | | '_ \ / _ \ '_ \ | ' /| '__/ _ \| |_| | | | \ \/ /
| |_| | |_) |  __/ | | || . \| | |  __/|  _| | |_| |>  < 
 \___/| .__/ \___|_| |_||_|\_\_|  \___||_| |_|\__,_/_/\_\
      |_|                                                 
""",
        style="bold cyan",
    )
    console.print(banner)

    table = Table(title="OpenKreflux Architecture Specs", show_header=True, header_style="bold magenta")
    table.add_column("Subsystem", style="cyan", width=22)
    table.add_column("Details", style="white")

    table.add_row("Version", f"v{__version__} (PEP 621)")
    table.add_row(
        "Supported Providers",
        "Featherless AI, OpenRouter, Neokens, Custom OpenAI-compatible endpoints",
    )
    table.add_row(
        "Failover Engine",
        "EWMA Latency Ranking + Exponential Capacity Backoff + Stream Resumption",
    )
    table.add_row(
        "Reasoning Ladders",
        "Low (Fast, ~10+ tok) | Medium (Balanced, ~150+ tok) | High (Deep, ~500+ tok) | Ultra (~1200+ tok, rigorous)",
    )
    table.add_row(
        "Verification Checks",
        "<think> tags, Math delimiter balance, Python AST syntax, Degenerate loop detection, Coherence score",
    )

    console.print(table)


@main.command(name="verify-trace")
@click.argument("filepath", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--target-level",
    "-l",
    type=click.Choice(["low", "medium", "high", "ultra"], case_sensitive=False),
    default=None,
    help="Target reasoning ladder depth to enforce",
)
@click.option(
    "--show-thoughts/--no-show-thoughts",
    default=False,
    help="Display extracted thinking trace content in a terminal panel",
)
def verify_trace(filepath: Path, target_level: Optional[str], show_thoughts: bool):
    """Verify and score a reasoning trace file."""
    trace_text = filepath.read_text(encoding="utf-8")
    verifier = ReasoningVerifier()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Evaluating reasoning trace...", total=None)
        result = verifier.verify(trace_text, target_ladder=target_level)

    # Result Header
    if result.is_valid:
        status_panel = Panel(
            Text(f"✔ VALID REASONING TRACE (Ladder: {result.ladder_level.value.upper()})", style="bold green"),
            border_style="green",
        )
    else:
        status_panel = Panel(
            Text(
                f"✖ VALIDATION FAILED (Observed: {result.ladder_level.value.upper()})",
                style="bold red",
            ),
            border_style="red",
        )
    console.print(status_panel)

    # Metrics Table
    metrics_table = Table(title="Trace Metrics", header_style="bold blue")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="bold yellow")

    metrics_table.add_row("Thinking Tokens", str(result.thinking_tokens))
    metrics_table.add_row("Solution Tokens", str(result.solution_tokens))
    metrics_table.add_row("Total Tokens", str(result.total_tokens))
    metrics_table.add_row(
        "Reasoning Density",
        f"{result.reasoning_density * 100:.1f}% ({'█' * int(result.reasoning_density * 20)}{'░' * (20 - int(result.reasoning_density * 20))})",
    )
    metrics_table.add_row("Coherence Score", f"{result.coherence_score:.2f} / 1.00")
    metrics_table.add_row("Self-Correction Detected", "Yes" if result.has_self_correction else "No")

    if result.math_validity is not None:
        math_str = "[green]Valid[/]" if result.math_validity else "[red]Syntax Error[/]"
        metrics_table.add_row("Math Syntax & Delimiters", math_str)
    if result.code_validity is not None:
        code_str = "[green]Valid[/]" if result.code_validity else "[red]Syntax Error[/]"
        metrics_table.add_row("Code AST Syntax", code_str)

    if result.meets_target_ladder is not None:
        target_str = "[green]Satisfied[/]" if result.meets_target_ladder else "[red]Unmet[/]"
        metrics_table.add_row(f"Meets Target ('{target_level}')", target_str)

    console.print(metrics_table)

    if result.issues:
        issue_table = Table(title="Issues & Diagnostics", header_style="bold red")
        issue_table.add_column("#", style="white", width=4)
        issue_table.add_column("Issue Description", style="red")
        for i, issue in enumerate(result.issues, 1):
            issue_table.add_row(str(i), issue)
        console.print(issue_table)

    if show_thoughts and result.extracted_thoughts:
        console.print(
            Panel(
                "\n\n".join(result.extracted_thoughts),
                title="[bold yellow]Extracted Thought Trace[/]",
                border_style="yellow",
            )
        )

    if not result.is_valid:
        sys.exit(1)


@main.command(name="benchmark")
@click.option("--model", "-m", default="kreflux-preview", help="Target model identifier")
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["featherless", "openrouter", "neokens", "mock"]),
    default="mock",
    help="Provider backend to benchmark ('mock' executes synthetic resilience benchmark)",
)
@click.option("--prompt", default=None, help="Custom prompt to benchmark")
@click.option("--iterations", "-i", default=2, help="Number of benchmark iterations")
def benchmark(model: str, provider: str, prompt: Optional[str], iterations: int):
    """Benchmark inference latency, TTFT, TPS, and failover speed."""
    console.print(f"[bold cyan]Starting benchmark for model:[/] [yellow]{model}[/] on [green]{provider}[/]")

    if provider == "mock":
        # Create simulated router with mock transport
        def mock_handler(request: httpx.Request) -> httpx.Response:
            url_str = str(request.url)
            if "featherless" in url_str:
                content = (
                    b"data: {\"choices\": [{\"delta\": {\"reasoning_content\": \"Let us reason through this step-by-step. \"}}]}\n\n"
                    b"data: {\"choices\": [{\"delta\": {\"reasoning_content\": \"Considering edge cases and formulas. \"}}]}\n\n"
                )
                def sse_gen():
                    yield content
                    raise httpx.StreamError("Simulated primary network drop")
                return httpx.Response(200, content=sse_gen())
            else:
                content = (
                    b"data: {\"choices\": [{\"delta\": {\"content\": \"Here is the verified solution. \"}}]}\n\n"
                    b"data: {\"choices\": [{\"delta\": {\"content\": \"Calculations hold.\", \"finish_reason\": \"stop\"}}]}\n\n"
                    b"data: [DONE]\n\n"
                )
                return httpx.Response(200, content=content)

        client = httpx.Client(transport=httpx.MockTransport(mock_handler))
        router = FlukoRouter(client=client)
        p1 = ProviderConfig(name="featherless", base_url="http://mock-featherless/v1", priority=1)
        p2 = ProviderConfig(name="openrouter", base_url="http://mock-openrouter/v1", priority=2)
        router.register_provider(p1)
        router.register_provider(p2)

        console.print("[dim]Simulating streaming chunks and failover resilience...[/]")
        metric = InferenceBenchmark(router).run_stream_benchmark(
            messages=[{"role": "user", "content": prompt or "Evaluate reasoning speed"}],
            model=model,
        )
        table = Table(title="Synthetic Benchmark Result", header_style="bold green")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold yellow")
        table.add_row("TTFT", f"{metric.ttft_ms:.1f} ms")
        table.add_row("TPS", f"{metric.tps:.1f} tokens/s")
        table.add_row("Reasoning Ratio", f"{metric.reasoning_token_ratio * 100:.1f}%")
        table.add_row("Failover Observed", "Yes" if metric.failover_occurred else "No")
        table.add_row("Failover Latency", f"{metric.failover_latency_ms:.1f} ms")
        table.add_row("Total Time", f"{metric.total_time_ms:.1f} ms")
        console.print(table)
        return

    # Real provider
    api_key = os.environ.get(f"{provider.upper()}_API_KEY")
    if not api_key:
        console.print(
            f"[bold red]Error:[/] Environment variable {provider.upper()}_API_KEY not set. Set it or use --provider mock."
        )
        sys.exit(1)

    router = FlukoRouter.create_default(**{f"{provider}_key": api_key})
    bench = InferenceBenchmark(router)

    prompts = [prompt] if prompt else None
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description=f"Benchmarking {iterations} iterations...", total=None)
        report = bench.run_suite(prompts=prompts, model=model, iterations_per_prompt=iterations)

    table = Table(title=f"Benchmark Report ({report.iterations} iterations)", header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold yellow")
    table.add_row("Mean TTFT", f"{report.mean_ttft_ms:.1f} ms")
    table.add_row("Median TTFT", f"{report.median_ttft_ms:.1f} ms")
    table.add_row("p95 TTFT", f"{report.p95_ttft_ms:.1f} ms")
    table.add_row("Mean TPS", f"{report.mean_tps:.1f} tok/s")
    table.add_row("Reasoning Token Ratio", f"{report.mean_reasoning_ratio * 100:.1f}%")
    table.add_row("Failovers Observed", str(report.failover_count))
    console.print(table)


if __name__ == "__main__":
    main()
