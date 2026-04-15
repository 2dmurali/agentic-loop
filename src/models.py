from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class Task(BaseModel):
    name: str
    min_reviews: int = 1
    max_reviews: int = 5
    pass_threshold: float | None = None
    worker_model: str | None = None
    reviewer_model: str | None = None
    output_filename: str = "output.md"
    worker_goal: str
    worker_instructions: str
    reviewer_criteria: str
    task_dir: Path


class ReviewResult(BaseModel):
    verdict: Literal["APPROVE", "REVISE"]
    score: int = Field(ge=0, le=10)
    comments: list[str]
    action_items: list[str] = []


class TaskResult(BaseModel):
    status: Literal["completed", "max_iterations_reached"]
    iterations: int
    final_score: int | None = None
