"""Pydantic output schemas for the Debate pipeline."""

from __future__ import annotations

from pydantic import BaseModel


class Argument(BaseModel):
    position: str          # "for" | "against"
    opening: str           # opening statement
    key_points: list[str]  # main arguments
    evidence: list[str]    # supporting facts / examples


class Rebuttal(BaseModel):
    position: str
    round: int
    counters: list[str]    # specific counter-arguments
    new_points: list[str]  # new arguments introduced


class Verdict(BaseModel):
    winner: str            # "for" | "against" | "draw"
    reasoning: str
    scores: dict[str, int] # {"for": N, "against": M} out of 10
    strongest_argument: str
    weakest_argument: str
    summary: str
