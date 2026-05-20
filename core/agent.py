"""Base Agent class — the unit of work in every pipeline."""

from __future__ import annotations

import json
import os
import time
from abc import ABC
from typing import Any, TYPE_CHECKING

MAX_CONTINUATIONS = 2  # retries when the model hits max_tokens mid-response

import anthropic
from pydantic import BaseModel
from rich.console import Console

if TYPE_CHECKING:
    from orchestrator.core.blackboard import Blackboard
    from orchestrator.core.tracer import Tracer

console = Console()


class TruncationError(RuntimeError):
    """Raised when the model stops because it hit max_tokens mid-response."""

    def __init__(self, agent_name: str, max_tokens: int, chars_so_far: int):
        super().__init__(
            f"{agent_name} hit max_tokens={max_tokens} after {chars_so_far} chars. "
            "Will retry with a continuation prompt."
        )
        self.partial: str = ""  # populated by the caller before raising


DEFAULT_MODEL = "claude-opus-4-6"

# Backend identifiers
BACKEND_ANTHROPIC = "anthropic"
BACKEND_OPENAI_COMPAT = "openai_compatible"  # Redpill, Groq, Ollama, etc.


class Agent(ABC):
    """
    Base class for all agents.

    Subclasses typically only need to set class-level attributes and override
    `build_prompt()`. Everything else — streaming, JSON parsing, token tracking,
    blackboard writes — is handled by the base class.

    Class attributes
    ----------------
    name          : Display name shown in logs and the tracer table.
    role          : One-line description of what this agent does.
    system_prompt : The system prompt sent to the model.
    model         : Model name (e.g. "claude-opus-4-6" or "minimax/minimax-m2.5").
    backend       : "anthropic" (default) or "openai_compatible".
    base_url      : Base URL for openai_compatible backends. Falls back to the
                    OPENAI_COMPAT_BASE_URL env var if not set on the class.
    api_key_env   : Name of the env var holding the API key for openai_compatible
                    backends. Defaults to OPENAI_COMPAT_API_KEY.
    output_schema : A Pydantic model class. If set, the agent's response is
                    parsed as JSON and validated against this schema.
    reads         : Blackboard keys this agent reads.
    writes        : Blackboard keys this agent writes.
    max_tokens       : Hard token cap for the response.
    use_thinking     : Whether to enable extended thinking (Anthropic only).
    timeout_seconds  : Cancel the agent after this many seconds. None = no limit.
    timeout_action   : What to do on timeout — "skip" (default, log and continue),
                       "raise" (propagate TimeoutError), or a fallback Agent instance
                       that runs in place of the timed-out agent.
    error_fallback   : Agent to run if this agent raises any exception. If None,
                       the exception propagates normally.
    """

    name: str = "Agent"
    role: str = ""
    system_prompt: str = ""
    model: str = DEFAULT_MODEL
    backend: str = BACKEND_ANTHROPIC
    base_url: str = ""          # openai_compatible only
    api_key_env: str = "OPENAI_COMPAT_API_KEY"
    output_schema: type[BaseModel] | None = None
    reads: list[str] = []
    writes: list[str] = []
    max_tokens: int = 8192
    use_thinking: bool = True
    timeout_seconds: float | None = None  # None = no timeout
    timeout_action: "str | Agent" = "skip"  # "skip", "raise", or a fallback Agent
    error_fallback: "Agent | None" = None  # agent to run if this one raises an exception

    # ------------------------------------------------------------------
    # Override in subclasses
    # ------------------------------------------------------------------

    def build_prompt(self, board: "Blackboard") -> str:
        """
        Build the user-turn prompt from the current blackboard state.
        Default implementation dumps all readable keys as JSON context.
        """
        relevant = {k: board.get(k) for k in self.reads if board.has(k)}
        if not relevant:
            return board.get("input", str) or ""
        return json.dumps(relevant, indent=2, default=str)

    def parse_output(self, raw: str) -> Any:
        """
        Parse the model's raw text response.
        If `output_schema` is set, strips markdown fences and validates JSON.
        """
        if self.output_schema is None:
            return raw.strip()

        clean = raw.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            # Remove opening fence (e.g. ```json)
            clean = "\n".join(lines[1:])
        if clean.endswith("```"):
            clean = clean[: clean.rfind("```")]

        clean = self._sanitize_json(clean.strip())
        data = json.loads(clean)
        data = self._normalize_keys(data)
        return self.output_schema(**data)

    @staticmethod
    def _normalize_keys(data: dict) -> dict:
        """
        Normalize top-level JSON keys to fix common model mistakes:
        - camelCase → snake_case:  componentSpecs → component_specs
        - Collapse multiple underscores: out_of__scope → out_of_scope
        - Strip leading/trailing underscores and spaces
        """
        import re

        def normalize(k: str) -> str:
            # camelCase → snake_case
            k = re.sub(r"([A-Z])", r"_\1", k).lower()
            # collapse multiple underscores
            k = re.sub(r"_+", "_", k)
            return k.strip("_ ")

        return {normalize(k): v for k, v in data.items()}

    @staticmethod
    def _sanitize_json(text: str) -> str:
        """
        Escape literal control characters inside JSON strings.

        Some models stream literal newlines/tabs inside JSON string values,
        which is invalid JSON. This walks the text character by character and
        escapes control characters that appear inside a string.

        Correctly handles escaped backslashes (\\) before quotes so that \\"
        is treated as an escaped backslash + closing quote, not an escaped quote.
        """
        result: list[str] = []
        in_string = False
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == '"':
                # Count consecutive backslashes immediately before this quote
                # in the result buffer to determine if the quote is escaped.
                # An odd count means the quote is escaped; even means it closes/opens.
                num_backslashes = 0
                j = len(result) - 1
                while j >= 0 and result[j] == "\\":
                    num_backslashes += 1
                    j -= 1
                if num_backslashes % 2 == 0:
                    in_string = not in_string
                result.append(ch)
            elif in_string:
                if ch == "\n":
                    result.append("\\n")
                elif ch == "\r":
                    result.append("\\r")
                elif ch == "\t":
                    result.append("\\t")
                else:
                    result.append(ch)
            else:
                result.append(ch)
            i += 1
        return "".join(result)

    def write_outputs(self, result: Any, board: "Blackboard") -> None:
        """
        Write parsed result to the blackboard.
        Default: write the full result to `writes[0]`.
        Override for agents that write to multiple keys.
        """
        if self.writes:
            board.set(self.writes[0], result, agent=self.name)

    # ------------------------------------------------------------------
    # Internal — not normally overridden
    # ------------------------------------------------------------------

    def _call_api(self, prompt: str) -> tuple[str, dict[str, int]]:
        """
        Stream a request to the configured backend.
        Returns (full_text, {"input_tokens": N, "output_tokens": M}).
        """
        console.print(f"\n[bold cyan]▶ {self.name}[/bold cyan]")

        if self.backend == BACKEND_ANTHROPIC:
            return self._call_anthropic(prompt)
        elif self.backend == BACKEND_OPENAI_COMPAT:
            return self._call_openai_compat(prompt)
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")

    def _call_anthropic(self, prompt: str) -> tuple[str, dict[str, int]]:
        """Anthropic streaming call with optional extended thinking."""
        client = anthropic.Anthropic()
        collected: list[str] = []

        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        if self.use_thinking:
            kwargs["thinking"] = {"type": "adaptive"}

        with client.messages.stream(**kwargs) as stream:
            for event in stream:
                if (
                    event.type == "content_block_delta"
                    and event.delta.type == "text_delta"
                ):
                    collected.append(event.delta.text)
                    console.print(event.delta.text, end="")
            final = stream.get_final_message()

        console.print()
        full_text = "".join(collected)

        if final.stop_reason == "max_tokens":
            err = TruncationError(self.name, self.max_tokens, len(full_text))
            err.partial = full_text
            raise err

        return full_text, {
            "input_tokens": final.usage.input_tokens,
            "output_tokens": final.usage.output_tokens,
        }

    def _call_openai_compat(self, prompt: str) -> tuple[str, dict[str, int]]:
        """OpenAI-compatible streaming call — works with Redpill, Groq, Ollama, etc."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required for openai_compatible backend. "
                "Run: pip install openai"
            )

        base_url = self.base_url or os.environ.get("OPENAI_COMPAT_BASE_URL", "")
        api_key = os.environ.get(self.api_key_env, "")

        if not base_url:
            raise ValueError(
                "base_url not set. Either set Agent.base_url or the "
                "OPENAI_COMPAT_BASE_URL environment variable."
            )
        if not api_key:
            raise ValueError(
                f"API key not found. Set the {self.api_key_env!r} environment variable."
            )

        client = OpenAI(base_url=base_url, api_key=api_key)
        collected: list[str] = []
        input_tokens = output_tokens = 0

        stream = client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=True,
            stream_options={"include_usage": True},
        )

        truncated = False
        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    collected.append(delta.content)
                    console.print(delta.content, end="")
                if chunk.choices[0].finish_reason == "length":
                    truncated = True
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens or 0
                output_tokens = chunk.usage.completion_tokens or 0

        console.print()
        full_text = "".join(collected)

        if truncated:
            err = TruncationError(self.name, self.max_tokens, len(full_text))
            err.partial = full_text
            raise err

        return full_text, {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    def run(self, board: "Blackboard", tracer: "Tracer | None" = None) -> None:
        """
        Full agent lifecycle:
          1. Build prompt from blackboard
          2. Call API (streaming)
          3. Parse and validate output
          4. Write results back to blackboard
          5. Notify tracer
        """
        if tracer:
            tracer.on_agent_start(self.name, self.model)

        start = time.time()
        retries = 0

        try:
            prompt = self.build_prompt(board)
            total_input_tokens = total_output_tokens = 0
            accumulated = ""

            for attempt in range(MAX_CONTINUATIONS + 1):
                usage: dict[str, int] = {}
                try:
                    raw, usage = self._call_api(prompt)
                    accumulated += raw
                    total_input_tokens += usage["input_tokens"]
                    total_output_tokens += usage["output_tokens"]
                    break  # clean finish_reason=stop — done
                except TruncationError as exc:
                    accumulated += exc.partial
                    total_output_tokens += usage.get("output_tokens", 0)
                    retries += 1
                    if attempt == MAX_CONTINUATIONS:
                        console.print(
                            f"\n[bold yellow]⚠ {self.name}: truncated after "
                            f"{MAX_CONTINUATIONS} continuations — parsing partial output[/bold yellow]"
                        )
                        break
                    console.print(
                        f"\n[yellow]↩ {self.name}: truncated, continuing "
                        f"(attempt {attempt + 2}/{MAX_CONTINUATIONS + 1})…[/yellow]"
                    )
                    prompt = (
                        "Continue the JSON exactly from where you stopped. "
                        "Do not repeat or summarise anything. "
                        f"Partial output so far:\n{exc.partial}"
                    )

            result = self.parse_output(accumulated)
            self.write_outputs(result, board)

            if tracer:
                tracer.on_agent_end(
                    self.name,
                    success=True,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    retries=retries,
                )

        except Exception as exc:
            if tracer:
                tracer.on_agent_end(
                    self.name,
                    success=False,
                    retries=retries,
                    error=str(exc),
                )
            raise

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
