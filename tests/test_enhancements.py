"""
Smoke tests for the three orchestrator enhancements:
  1. Truncation detection + auto-continuation
  2. Agent timeout (skip / raise / fallback)
  3. Schema validation feedback loop
  4. Error-based fallback

All tests mock _call_api so no API key is required.
"""

import json
import time
from unittest.mock import patch

from pydantic import BaseModel

from orchestrator.core.agent import Agent, TruncationError, BACKEND_OPENAI_COMPAT
from orchestrator.core.blackboard import Blackboard
from orchestrator.core.pipeline import Pipeline
from orchestrator.core.tracer import Tracer


# ---------------------------------------------------------------------------
# Shared schema
# ---------------------------------------------------------------------------

class SimpleOutput(BaseModel):
    message: str
    value: int


# ---------------------------------------------------------------------------
# Helper: build a minimal agent without calling super().__init__
# ---------------------------------------------------------------------------

def make_agent(**kwargs):
    """Dynamically create an Agent subclass with the given attributes."""
    attrs = {
        "name": "TestAgent",
        "role": "test",
        "system_prompt": "test",
        "backend": BACKEND_OPENAI_COMPAT,
        "output_schema": SimpleOutput,
        "reads": [],
        "writes": ["result"],
        **kwargs,
    }
    return type("DynAgent", (Agent,), attrs)()


VALID_JSON = json.dumps({"message": "hello", "value": 42})


# ---------------------------------------------------------------------------
# Test 1: Truncation continuation
# ---------------------------------------------------------------------------

def test_truncation_continuation():
    print("\n[1] Truncation continuation...", end=" ")

    agent = make_agent(name="TruncAgent")

    call_count = 0

    def fake_call_api(prompt):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Simulate truncation on first call
            err = TruncationError("TruncAgent", 8192, 10)
            err.partial = '{"message": "hel'
            raise err
        # Second call: continuation returns the rest
        return 'lo", "value": 42}', {"input_tokens": 5, "output_tokens": 5}

    board = Blackboard()
    with patch.object(agent, "_call_api", side_effect=fake_call_api):
        agent.run(board)

    assert call_count == 2, f"expected 2 calls, got {call_count}"
    result = board.get("result")
    assert result.message == "hello"
    assert result.value == 42
    print("PASSED")


# ---------------------------------------------------------------------------
# Test 2: Timeout — skip
# ---------------------------------------------------------------------------

def test_timeout_skip():
    print("[2] Timeout (skip)...", end=" ")

    agent = make_agent(name="SlowAgent", timeout_seconds=0.1, timeout_action="skip")

    def slow_call_api(prompt):
        time.sleep(2)
        return VALID_JSON, {"input_tokens": 5, "output_tokens": 5}

    with patch.object(agent, "_call_api", side_effect=slow_call_api):
        pipeline = Pipeline().then(agent)
        result_board = pipeline.run()  # should NOT raise

    assert result_board.get("result") is None, "result should be empty after skip"
    print("PASSED")


# ---------------------------------------------------------------------------
# Test 3: Timeout — raise
# ---------------------------------------------------------------------------

def test_timeout_raise():
    print("[3] Timeout (raise)...", end=" ")

    agent = make_agent(name="SlowAgent2", timeout_seconds=0.1, timeout_action="raise")

    def slow_call_api(prompt):
        time.sleep(2)
        return VALID_JSON, {"input_tokens": 5, "output_tokens": 5}

    raised = False
    with patch.object(agent, "_call_api", side_effect=slow_call_api):
        try:
            pipeline = Pipeline().then(agent)
            pipeline.run()
        except TimeoutError:
            raised = True

    assert raised, "expected TimeoutError"
    print("PASSED")


# ---------------------------------------------------------------------------
# Test 4: Timeout — fallback agent
# ---------------------------------------------------------------------------

def test_timeout_fallback():
    print("[4] Timeout (fallback agent)...", end=" ")

    fallback = make_agent(name="FallbackAgent", writes=["result"])

    def fallback_call_api(prompt):
        return VALID_JSON, {"input_tokens": 2, "output_tokens": 2}

    with patch.object(fallback, "_call_api", side_effect=fallback_call_api):
        main_agent = make_agent(
            name="SlowMain",
            timeout_seconds=0.1,
            timeout_action=fallback,
        )

        def slow_call_api(prompt):
            time.sleep(2)
            return VALID_JSON, {"input_tokens": 5, "output_tokens": 5}

        with patch.object(main_agent, "_call_api", side_effect=slow_call_api):
            pipeline = Pipeline().then(main_agent)
            result_board = pipeline.run()

    result = result_board.get("result")
    assert result is not None, "fallback should have written result"
    assert result.message == "hello"
    print("PASSED")


# ---------------------------------------------------------------------------
# Test 5: Schema validation feedback loop
# ---------------------------------------------------------------------------

def test_validation_feedback_loop():
    print("[5] Validation feedback loop...", end=" ")

    agent = make_agent(name="ValidationAgent", max_validation_retries=2)

    call_count = 0

    def fake_call_api(prompt):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call: invalid JSON (missing required field)
            return '{"message": "hello"}', {"input_tokens": 5, "output_tokens": 5}
        # Second call (after feedback): valid JSON
        return VALID_JSON, {"input_tokens": 5, "output_tokens": 5}

    board = Blackboard()
    with patch.object(agent, "_call_api", side_effect=fake_call_api):
        agent.run(board)

    assert call_count == 2, f"expected 2 calls, got {call_count}"
    result = board.get("result")
    assert result.value == 42
    print("PASSED")


# ---------------------------------------------------------------------------
# Test 6: Error-based fallback
# ---------------------------------------------------------------------------

def test_error_fallback():
    print("[6] Error-based fallback...", end=" ")

    fallback = make_agent(name="ErrorFallback", writes=["result"])

    def fallback_call_api(prompt):
        return VALID_JSON, {"input_tokens": 2, "output_tokens": 2}

    with patch.object(fallback, "_call_api", side_effect=fallback_call_api):
        main_agent = make_agent(name="BrokenAgent", error_fallback=fallback)

        def broken_call_api(prompt):
            raise RuntimeError("Simulated API failure")

        with patch.object(main_agent, "_call_api", side_effect=broken_call_api):
            pipeline = Pipeline().then(main_agent)
            result_board = pipeline.run()

    result = result_board.get("result")
    assert result is not None, "fallback should have written result"
    assert result.message == "hello"
    print("PASSED")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_truncation_continuation,
        test_timeout_skip,
        test_timeout_raise,
        test_timeout_fallback,
        test_validation_feedback_loop,
        test_error_fallback,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAILED — {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)
