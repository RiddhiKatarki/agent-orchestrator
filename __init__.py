"""
orchestrator — a lightweight multi-agent pipeline framework.

Quick start
-----------
from orchestrator import Pipeline, Tracer
from orchestrator.core.agent import Agent

class MyAgent(Agent):
    name = "MyAgent"
    system_prompt = "You are helpful."
    reads = ["input"]
    writes = ["output"]

    def build_prompt(self, board):
        return board.get("input", str)

pipeline = Pipeline(name="Demo").then(MyAgent())
board = pipeline.run("Hello, world!")
print(board.get("output"))
"""

from orchestrator.core.agent import Agent
from orchestrator.core.blackboard import Blackboard
from orchestrator.core.pipeline import Pipeline
from orchestrator.core.tracer import Tracer

__all__ = ["Agent", "Blackboard", "Pipeline", "Tracer"]
