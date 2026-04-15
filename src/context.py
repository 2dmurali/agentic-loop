from __future__ import annotations
import json
from pathlib import Path
from src.models import Task, Message, ReviewResult


class ContextAccumulator:
    def __init__(self, task: Task):
        self.task = task
        self.worker_system_prompt = self._build_worker_system_prompt(task)
        self.reviewer_system_prompt = (
            f"You are a thorough reviewer agent.\n\n## Review Criteria\n{task.reviewer_criteria}\n\n"
            "You MUST respond with valid JSON in this exact format:\n"
            '{"verdict": "APPROVE" or "REVISE", "score": 0-10, "comments": ["..."], "action_items": ["..."]}\n'
            "Only output the JSON, nothing else."
        )
        self._worker_messages: list[Message] = [
            Message(role="user", content=f"## Task Goal\n{task.worker_goal}")
        ]
        self._reviewer_messages: list[Message] = []
        self._current_iteration = 0

    @staticmethod
    def _build_worker_system_prompt(task: Task) -> str:
        base = f"You are a skilled worker agent.\n\n## Instructions\n{task.worker_instructions}"
        if not task.output_filename.endswith(".md"):
            base += (
                f"\n\nIMPORTANT: Output ONLY the file content for `{task.output_filename}`. "
                "Do not wrap your response in markdown fenced code blocks (no ```). "
                "Do not include any explanation, commentary, or text before or after the content. "
                f"Your entire response will be saved directly as `{task.output_filename}`."
            )
        return base

    def get_worker_messages(self) -> list[Message]:
        """Return full worker conversation history."""
        return list(self._worker_messages)

    def get_reviewer_messages(self) -> list[Message]:
        """Return full reviewer conversation history."""
        return list(self._reviewer_messages)

    def add_worker_output(self, iteration: int, content: str) -> None:
        """Append worker's output to both worker and reviewer context."""
        self._current_iteration = iteration
        self._worker_messages.append(Message(role="assistant", content=content))
        self._reviewer_messages.append(
            Message(role="user", content=f"## Worker Output (Iteration {iteration})\n{content}")
        )

    def add_review(self, iteration: int, review: ReviewResult) -> None:
        """Append reviewer feedback to both contexts."""
        feedback = f"## Reviewer Feedback (Iteration {iteration})\n"
        feedback += f"Verdict: {review.verdict} | Score: {review.score}/10\n\n"
        if review.comments:
            feedback += "### Comments\n" + "\n".join(f"- {c}" for c in review.comments) + "\n\n"
        if review.action_items:
            feedback += "### Action Items\n" + "\n".join(f"- {a}" for a in review.action_items)

        self._reviewer_messages.append(Message(role="assistant", content=json.dumps(review.model_dump())))
        self._worker_messages.append(Message(role="user", content=feedback))

    def inject_current_output(self, content: str, iteration: int) -> None:
        """Inject current deliverable content into worker context for revision."""
        self._worker_messages.append(
            Message(
                role="user",
                content=(
                    f"## Current Output (before iteration {iteration})\n"
                    "Here is your current output. Revise it based on the reviewer feedback above.\n\n"
                    f"{content}"
                ),
            )
        )

    @property
    def current_iteration(self) -> int:
        return self._current_iteration

    def to_dict(self) -> dict:
        """Serialize context for saving/resuming."""
        return {
            "task_name": self.task.name,
            "current_iteration": self._current_iteration,
            "worker_messages": [m.model_dump() for m in self._worker_messages],
            "reviewer_messages": [m.model_dump() for m in self._reviewer_messages],
        }

    def save(self, path: Path) -> None:
        """Save context to JSON file."""
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path, task: Task) -> ContextAccumulator:
        """Load context from JSON file."""
        data = json.loads(path.read_text())
        ctx = cls(task)
        ctx._current_iteration = data["current_iteration"]
        ctx._worker_messages = [Message(**m) for m in data["worker_messages"]]
        ctx._reviewer_messages = [Message(**m) for m in data["reviewer_messages"]]
        return ctx
