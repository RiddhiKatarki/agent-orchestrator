"""Tracer — records every agent run, tracks tokens and cost, renders a summary."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text

console = Console()

# Approximate pricing per million tokens (input / output).
# Update these as Anthropic publishes new prices.
_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-6":    (15.0, 75.0),
    "claude-sonnet-4-6":  (3.0,  15.0),
    "claude-haiku-4-5":   (0.80, 4.0),
}

_DEFAULT_PRICING = (15.0, 75.0)  # fallback for unknown models


def _cost(model: str, input_tokens: int, output_tokens: int) -> float:
    in_price, out_price = _PRICING.get(model, _DEFAULT_PRICING)
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


@dataclass
class AgentSpan:
    agent_name: str
    model: str
    started_at: float = field(default_factory=time.time)
    ended_at: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    retries: int = 0
    success: bool = True
    error: str = ""

    @property
    def elapsed(self) -> float:
        return self.ended_at - self.started_at

    @property
    def cost_usd(self) -> float:
        return _cost(self.model, self.input_tokens, self.output_tokens)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class Tracer:
    """
    Attach a Tracer to a Pipeline to record every agent call.

    After the run, call .render() to print a Rich summary table.
    """

    def __init__(self) -> None:
        self._spans: list[AgentSpan] = []
        self._active: dict[str, AgentSpan] = {}  # agent_name -> open span

    # ------------------------------------------------------------------
    # Hooks called by Agent.run()
    # ------------------------------------------------------------------

    def on_agent_start(self, agent_name: str, model: str) -> None:
        span = AgentSpan(agent_name=agent_name, model=model)
        self._active[agent_name] = span

    def on_agent_end(
        self,
        agent_name: str,
        *,
        success: bool = True,
        input_tokens: int = 0,
        output_tokens: int = 0,
        retries: int = 0,
        error: str = "",
    ) -> None:
        span = self._active.pop(agent_name, None)
        if span is None:
            return
        span.ended_at = time.time()
        span.success = success
        span.input_tokens = input_tokens
        span.output_tokens = output_tokens
        span.retries = retries
        span.error = error
        self._spans.append(span)

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------

    @property
    def spans(self) -> list[AgentSpan]:
        return list(self._spans)

    def total_cost(self) -> float:
        return sum(s.cost_usd for s in self._spans)

    def total_tokens(self) -> int:
        return sum(s.total_tokens for s in self._spans)

    def total_elapsed(self) -> float:
        if not self._spans:
            return 0.0
        start = min(s.started_at for s in self._spans)
        end = max(s.ended_at for s in self._spans)
        return end - start

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> None:
        if not self._spans:
            return

        table = Table(title="Pipeline Execution Summary", show_footer=True)
        table.add_column("Agent", style="bold")
        table.add_column("Status")
        table.add_column("Tokens", justify="right", footer=f"{self.total_tokens():,}")
        table.add_column("Cost (USD)", justify="right", footer=f"${self.total_cost():.4f}")
        table.add_column("Latency", justify="right", footer=f"{self.total_elapsed():.1f}s")
        table.add_column("Retries", justify="right")

        for span in self._spans:
            status = Text("✅ Pass", style="green") if span.success else Text("❌ Fail", style="red")
            retries = str(span.retries) if span.retries else "-"
            table.add_row(
                span.agent_name,
                status,
                f"{span.total_tokens:,}",
                f"${span.cost_usd:.4f}",
                f"{span.elapsed:.1f}s",
                retries,
            )

        console.print()
        console.print(table)

    def export(self) -> list[dict[str, Any]]:
        return [
            {
                "agent": s.agent_name,
                "model": s.model,
                "success": s.success,
                "input_tokens": s.input_tokens,
                "output_tokens": s.output_tokens,
                "cost_usd": s.cost_usd,
                "elapsed_s": s.elapsed,
                "retries": s.retries,
                "error": s.error,
            }
            for s in self._spans
        ]
