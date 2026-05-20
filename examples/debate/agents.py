"""Debate agents — three debaters and a judge, all on the orchestrator framework."""

from __future__ import annotations

from orchestrator.core.agent import Agent
from orchestrator.core.blackboard import Blackboard
from orchestrator.examples.debate.models import Argument, Rebuttal, Verdict


# ──────────────────────────────────────────────
# Opening statement agents
# ──────────────────────────────────────────────

class ProponentAgent(Agent):
    name = "Proponent"
    role = "Argues FOR the proposition"
    system_prompt = """You are an expert debater arguing FOR a proposition.
Build a compelling, evidence-based opening argument.
Be persuasive but intellectually honest. Use concrete examples.

Respond ONLY with valid JSON:
{
  "position": "for",
  "opening": "one powerful opening paragraph",
  "key_points": ["Point 1", "Point 2", "Point 3"],
  "evidence": ["Specific fact or example supporting your case", ...]
}"""
    output_schema = Argument
    reads = ["topic"]
    writes = ["for_argument"]

    def build_prompt(self, board: Blackboard) -> str:
        topic = board.get("topic", str)
        return f'Write a compelling opening argument FOR this proposition:\n\n"{topic}"'

    def write_outputs(self, result: Argument, board: Blackboard) -> None:
        board.set("for_argument", result, agent=self.name)


class OpponentAgent(Agent):
    name = "Opponent"
    role = "Argues AGAINST the proposition"
    system_prompt = """You are an expert debater arguing AGAINST a proposition.
Build a compelling, evidence-based opening argument.
Be persuasive but intellectually honest. Use concrete examples.

Respond ONLY with valid JSON:
{
  "position": "against",
  "opening": "one powerful opening paragraph",
  "key_points": ["Point 1", "Point 2", "Point 3"],
  "evidence": ["Specific fact or example supporting your case", ...]
}"""
    output_schema = Argument
    reads = ["topic"]
    writes = ["against_argument"]

    def build_prompt(self, board: Blackboard) -> str:
        topic = board.get("topic", str)
        return f'Write a compelling opening argument AGAINST this proposition:\n\n"{topic}"'

    def write_outputs(self, result: Argument, board: Blackboard) -> None:
        board.set("against_argument", result, agent=self.name)


# ──────────────────────────────────────────────
# Rebuttal agents (used inside the loop)
# ──────────────────────────────────────────────

class ProponentRebuttalAgent(Agent):
    name = "Proponent"
    role = "Rebuts the opponent's arguments (FOR side)"
    system_prompt = """You are an expert debater arguing FOR a proposition.
You have heard the opponent's arguments. Now rebut them specifically and introduce new supporting points.

Respond ONLY with valid JSON:
{
  "position": "for",
  "round": <current round number>,
  "counters": ["Specific rebuttal to opponent point", ...],
  "new_points": ["New argument not yet raised", ...]
}"""
    output_schema = Rebuttal
    reads = ["topic", "against_argument", "against_rebuttal", "_loop_iteration"]
    writes = ["for_rebuttal"]

    def build_prompt(self, board: Blackboard) -> str:
        topic = board.get("topic", str)
        round_num = (board.get("_loop_iteration") or 0) + 1

        against_arg = board.get("against_argument")
        against_reb = board.get("against_rebuttal")

        opponent_content = ""
        if against_arg:
            opponent_content += (
                f"\nOpponent's opening:\n{against_arg.opening}\n"
                f"Key points: {against_arg.key_points}\n"
            )
        if against_reb:
            opponent_content += (
                f"\nOpponent's last rebuttal:\n"
                f"Counters: {against_reb.counters}\n"
                f"New points: {against_reb.new_points}\n"
            )

        return (
            f'Debate round {round_num}. Proposition: "{topic}"\n'
            f"You are arguing FOR.\n\n"
            f"Opponent's arguments so far:{opponent_content}\n"
            f"Now rebut their points and strengthen your case."
        )

    def write_outputs(self, result: Rebuttal, board: Blackboard) -> None:
        board.set("for_rebuttal", result, agent=self.name)


class OpponentRebuttalAgent(Agent):
    name = "Opponent"
    role = "Rebuts the proponent's arguments (AGAINST side)"
    system_prompt = """You are an expert debater arguing AGAINST a proposition.
You have heard the proponent's arguments. Now rebut them specifically and introduce new points.

Respond ONLY with valid JSON:
{
  "position": "against",
  "round": <current round number>,
  "counters": ["Specific rebuttal to proponent point", ...],
  "new_points": ["New argument not yet raised", ...]
}"""
    output_schema = Rebuttal
    reads = ["topic", "for_argument", "for_rebuttal", "_loop_iteration"]
    writes = ["against_rebuttal"]

    def build_prompt(self, board: Blackboard) -> str:
        topic = board.get("topic", str)
        round_num = (board.get("_loop_iteration") or 0) + 1

        for_arg = board.get("for_argument")
        for_reb = board.get("for_rebuttal")

        proponent_content = ""
        if for_arg:
            proponent_content += (
                f"\nProponent's opening:\n{for_arg.opening}\n"
                f"Key points: {for_arg.key_points}\n"
            )
        if for_reb:
            proponent_content += (
                f"\nProponent's last rebuttal:\n"
                f"Counters: {for_reb.counters}\n"
                f"New points: {for_reb.new_points}\n"
            )

        return (
            f'Debate round {round_num}. Proposition: "{topic}"\n'
            f"You are arguing AGAINST.\n\n"
            f"Proponent's arguments so far:{proponent_content}\n"
            f"Now rebut their points and strengthen your case."
        )

    def write_outputs(self, result: Rebuttal, board: Blackboard) -> None:
        board.set("against_rebuttal", result, agent=self.name)


# ──────────────────────────────────────────────
# Judge
# ──────────────────────────────────────────────

class JudgeAgent(Agent):
    name = "Judge"
    role = "Evaluates both sides and delivers a verdict"
    system_prompt = """You are an impartial debate judge. Evaluate both sides fairly based on:
- Logic and coherence of arguments
- Quality of evidence
- Effectiveness of rebuttals
- Overall persuasiveness

Respond ONLY with valid JSON:
{
  "winner": "for" or "against" or "draw",
  "reasoning": "detailed explanation of the verdict",
  "scores": {"for": <0-10>, "against": <0-10>},
  "strongest_argument": "the single most compelling argument made in the debate",
  "weakest_argument": "the weakest or most flawed argument made",
  "summary": "brief neutral summary of the debate"
}"""
    output_schema = Verdict
    reads = ["topic", "for_argument", "against_argument", "for_rebuttal", "against_rebuttal"]
    writes = ["verdict"]

    def build_prompt(self, board: Blackboard) -> str:
        topic = board.get("topic", str)
        for_arg = board.get("for_argument")
        against_arg = board.get("against_argument")
        for_reb = board.get("for_rebuttal")
        against_reb = board.get("against_rebuttal")

        sections = [f'Debate proposition: "{topic}"\n']

        if for_arg:
            sections.append(
                f"=== FOR (opening) ===\n{for_arg.opening}\n"
                f"Key points: {for_arg.key_points}\n"
                f"Evidence: {for_arg.evidence}\n"
            )
        if against_arg:
            sections.append(
                f"=== AGAINST (opening) ===\n{against_arg.opening}\n"
                f"Key points: {against_arg.key_points}\n"
                f"Evidence: {against_arg.evidence}\n"
            )
        if for_reb:
            sections.append(
                f"=== FOR (rebuttals) ===\n"
                f"Counters: {for_reb.counters}\n"
                f"New points: {for_reb.new_points}\n"
            )
        if against_reb:
            sections.append(
                f"=== AGAINST (rebuttals) ===\n"
                f"Counters: {against_reb.counters}\n"
                f"New points: {against_reb.new_points}\n"
            )

        sections.append("Evaluate both sides and deliver your verdict.")
        return "\n".join(sections)
