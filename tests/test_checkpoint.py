"""
Tests for pipeline checkpointing and resume.

Uses the same mock pattern as test_enhancements.py — no real API calls.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from unittest.mock import patch

from orchestrator.core.agent import Agent
from orchestrator.core.checkpoint import FileCheckpointStore
from orchestrator.core.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_agent(name: str, reads: list, writes: list, response: str):
    """Create a minimal Agent subclass whose _call_api returns a fixed string."""
    cls = type(name, (Agent,), {
        "name": name,
        "role": name,
        "system_prompt": "",
        "reads": reads,
        "writes": writes,
        "use_thinking": False,
        "output_schema": None,
    })
    instance = cls()
    instance._call_api = lambda prompt: (response, {"input_tokens": 10, "output_tokens": 10})
    return instance


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCheckpoint(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = FileCheckpointStore(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    # ------------------------------------------------------------------ #
    # 1. Checkpoint is written after each node and cleared on completion  #
    # ------------------------------------------------------------------ #
    def test_checkpoint_written_and_cleared_on_success(self):
        agent_a = make_agent("A", reads=[], writes=["a_out"], response="result_a")
        agent_b = make_agent("B", reads=["a_out"], writes=["b_out"], response="result_b")

        pipeline = Pipeline(name="Test").then(agent_a).then(agent_b)
        pipeline.run(
            initial={"input": "go"},
            run_id="run-1",
            checkpoint_store=self.store,
        )

        # Checkpoint deleted after clean completion
        self.assertIsNone(self.store.load("run-1"),
                          "Checkpoint should be deleted after successful run")

    # ------------------------------------------------------------------ #
    # 2. Simulated crash: checkpoint left at node 1                       #
    # ------------------------------------------------------------------ #
    def test_resume_skips_completed_nodes(self):
        run_tracker = []

        def make_tracked_agent(name, reads, writes, response):
            agent = make_agent(name, reads, writes, response)
            original = agent._call_api
            def tracked(prompt):
                run_tracker.append(name)
                return original(prompt)
            agent._call_api = tracked
            return agent

        agent_a = make_tracked_agent("A", reads=[], writes=["a_out"], response="from_a")
        agent_b = make_tracked_agent("B", reads=[], writes=["b_out"], response="from_b")

        pipeline = Pipeline(name="Test").then(agent_a).then(agent_b)

        # Simulate crash after node 0 by manually saving a checkpoint at index 1
        from orchestrator.core.blackboard import Blackboard
        partial_board = Blackboard({"input": "go", "a_out": "from_a"})
        self.store.save("run-crash", node_index=1, board=partial_board)

        # Resume — only agent B should run
        result = pipeline.run(
            run_id="run-crash",
            checkpoint_store=self.store,
            resume=True,
        )

        self.assertNotIn("A", run_tracker, "Agent A should have been skipped on resume")
        self.assertIn("B", run_tracker, "Agent B should have run on resume")
        self.assertEqual(result.get("b_out"), "from_b")

    # ------------------------------------------------------------------ #
    # 3. Resume with no existing checkpoint starts from scratch           #
    # ------------------------------------------------------------------ #
    def test_resume_with_no_checkpoint_runs_full_pipeline(self):
        run_tracker = []

        def make_tracked(name):
            agent = make_agent(name, reads=[], writes=[f"{name}_out"], response=f"val_{name}")
            original = agent._call_api
            def tracked(prompt):
                run_tracker.append(name)
                return original(prompt)
            agent._call_api = tracked
            return agent

        pipeline = Pipeline(name="Test").then(make_tracked("X")).then(make_tracked("Y"))
        pipeline.run(
            run_id="nonexistent-run",
            checkpoint_store=self.store,
            resume=True,
        )

        self.assertEqual(run_tracker, ["X", "Y"],
                         "All agents should run when no checkpoint exists")

    # ------------------------------------------------------------------ #
    # 4. list_runs returns correct metadata                               #
    # ------------------------------------------------------------------ #
    def test_list_runs(self):
        from orchestrator.core.blackboard import Blackboard
        b = Blackboard({"key": "value"})
        self.store.save("run-alpha", 2, b)
        self.store.save("run-beta", 1, b)

        runs = self.store.list_runs()
        run_ids = [r["run_id"] for r in runs]
        self.assertIn("run-alpha", run_ids)
        self.assertIn("run-beta", run_ids)

    # ------------------------------------------------------------------ #
    # 5. Checkpoint survives a Pydantic model value on the board          #
    # ------------------------------------------------------------------ #
    def test_checkpoint_serialises_pydantic_values(self):
        from pydantic import BaseModel

        class Output(BaseModel):
            summary: str
            score: int

        from orchestrator.core.blackboard import Blackboard
        b = Blackboard({"result": Output(summary="hello", score=42)})
        self.store.save("run-pydantic", 1, b)

        _, snapshot = self.store.load("run-pydantic")
        # Pydantic model should be serialised as a dict
        self.assertEqual(snapshot["result"]["summary"], "hello")
        self.assertEqual(snapshot["result"]["score"], 42)


if __name__ == "__main__":
    unittest.main(verbosity=2)
