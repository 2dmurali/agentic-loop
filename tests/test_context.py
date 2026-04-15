from __future__ import annotations

import json
from pathlib import Path

from src.context import ContextAccumulator
from src.models import Message, ReviewResult, Task


def _make_task(**overrides) -> Task:
    defaults = dict(
        name="test-task",
        worker_goal="Do the thing",
        worker_instructions="Do it well",
        reviewer_criteria="Check it",
        task_dir=Path("/tmp/test-task"),
    )
    defaults.update(overrides)
    return Task(**defaults)


class TestContextAccumulator:
    def test_initial_state(self):
        task = _make_task()
        ctx = ContextAccumulator(task)
        messages = ctx.get_worker_messages()
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert "Do the thing" in messages[0].content
        assert ctx.get_reviewer_messages() == []
        assert ctx.current_iteration == 0

    def test_add_worker_output(self):
        task = _make_task()
        ctx = ContextAccumulator(task)
        ctx.add_worker_output(1, "Here is my work")

        worker_msgs = ctx.get_worker_messages()
        assert len(worker_msgs) == 2
        assert worker_msgs[1].role == "assistant"
        assert worker_msgs[1].content == "Here is my work"

        reviewer_msgs = ctx.get_reviewer_messages()
        assert len(reviewer_msgs) == 1
        assert reviewer_msgs[0].role == "user"
        assert "Here is my work" in reviewer_msgs[0].content
        assert ctx.current_iteration == 1

    def test_add_review(self):
        task = _make_task()
        ctx = ContextAccumulator(task)
        ctx.add_worker_output(1, "Work v1")

        review = ReviewResult(
            verdict="REVISE",
            score=5,
            comments=["Needs improvement"],
            action_items=["Fix edge case"],
        )
        ctx.add_review(1, review)

        worker_msgs = ctx.get_worker_messages()
        assert len(worker_msgs) == 3  # goal + worker output + review feedback
        assert worker_msgs[2].role == "user"
        assert "REVISE" in worker_msgs[2].content
        assert "Needs improvement" in worker_msgs[2].content

        reviewer_msgs = ctx.get_reviewer_messages()
        assert len(reviewer_msgs) == 2  # worker output + review
        assert reviewer_msgs[1].role == "assistant"

    def test_full_iteration_context_grows(self):
        task = _make_task()
        ctx = ContextAccumulator(task)

        # Iteration 1
        ctx.add_worker_output(1, "Work v1")
        ctx.add_review(1, ReviewResult(verdict="REVISE", score=4, comments=["Bad"]))

        # Iteration 2
        ctx.add_worker_output(2, "Work v2")
        ctx.add_review(2, ReviewResult(verdict="APPROVE", score=9, comments=["Good"]))

        worker_msgs = ctx.get_worker_messages()
        # goal + work1 + review1 + work2 + review2
        assert len(worker_msgs) == 5

        reviewer_msgs = ctx.get_reviewer_messages()
        # work1 + review1 + work2 + review2
        assert len(reviewer_msgs) == 4

    def test_serialization_roundtrip(self, tmp_path: Path):
        task = _make_task(task_dir=tmp_path)
        ctx = ContextAccumulator(task)
        ctx.add_worker_output(1, "Work v1")
        ctx.add_review(1, ReviewResult(verdict="REVISE", score=5, comments=["Fix it"]))

        save_path = tmp_path / "context.json"
        ctx.save(save_path)

        loaded = ContextAccumulator.load(save_path, task)
        assert loaded.current_iteration == 1
        assert len(loaded.get_worker_messages()) == len(ctx.get_worker_messages())
        assert len(loaded.get_reviewer_messages()) == len(ctx.get_reviewer_messages())

    def test_system_prompts(self):
        task = _make_task(
            worker_instructions="Be precise",
            reviewer_criteria="Check accuracy",
        )
        ctx = ContextAccumulator(task)
        assert "Be precise" in ctx.worker_system_prompt
        assert "Check accuracy" in ctx.reviewer_system_prompt
        assert "JSON" in ctx.reviewer_system_prompt

    def test_code_task_system_prompt(self):
        task = _make_task(output_filename="index.html")
        ctx = ContextAccumulator(task)
        assert "ONLY the file content" in ctx.worker_system_prompt
        assert "index.html" in ctx.worker_system_prompt
        assert "Do not wrap" in ctx.worker_system_prompt

    def test_text_task_no_file_instruction(self):
        task = _make_task(output_filename="output.md")
        ctx = ContextAccumulator(task)
        assert "ONLY the file content" not in ctx.worker_system_prompt

    def test_inject_current_output(self):
        task = _make_task()
        ctx = ContextAccumulator(task)
        ctx.add_worker_output(1, "Work v1")
        ctx.add_review(1, ReviewResult(verdict="REVISE", score=5, comments=["Fix it"]))

        ctx.inject_current_output("Current content here", 2)

        worker_msgs = ctx.get_worker_messages()
        last_msg = worker_msgs[-1]
        assert last_msg.role == "user"
        assert "Current content here" in last_msg.content
        assert "iteration 2" in last_msg.content

    def test_messages_are_copies(self):
        task = _make_task()
        ctx = ContextAccumulator(task)
        msgs = ctx.get_worker_messages()
        msgs.append(Message(role="user", content="extra"))
        assert len(ctx.get_worker_messages()) == 1  # original unchanged
