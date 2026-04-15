from __future__ import annotations

import pytest
from pathlib import Path

from src.task_loader import load_task, load_all_tasks, _parse_sections


class TestParseSections:
    def test_basic_sections(self):
        content = "## Goal\nDo something\n\n## Instructions\nStep 1\nStep 2\n"
        sections = _parse_sections(content)
        assert "Goal" in sections
        assert "Instructions" in sections
        assert sections["Goal"] == "Do something"
        assert "Step 1" in sections["Instructions"]

    def test_empty_content(self):
        assert _parse_sections("") == {}

    def test_no_headings(self):
        assert _parse_sections("Just some text\nwith no headings") == {}

    def test_content_before_first_heading(self):
        content = "Preamble text\n\n## Heading\nBody"
        sections = _parse_sections(content)
        assert len(sections) == 1
        assert sections["Heading"] == "Body"


class TestLoadTask:
    def test_load_valid_task(self, tmp_path: Path):
        task_dir = tmp_path / "my-task"
        task_dir.mkdir()
        (task_dir / "task.md").write_text(
            "---\nname: Test Task\nmin_reviews: 2\nmax_reviews: 3\n---\n\n"
            "## Worker Goal\nBuild something\n\n"
            "## Worker Instructions\nDo it well\n\n"
            "## Reviewer Criteria\nCheck quality\n"
        )
        task = load_task(task_dir)
        assert task.name == "Test Task"
        assert task.min_reviews == 2
        assert task.max_reviews == 3
        assert task.worker_goal == "Build something"
        assert task.worker_instructions == "Do it well"
        assert task.reviewer_criteria == "Check quality"
        assert task.task_dir == task_dir

    def test_missing_section_raises(self, tmp_path: Path):
        task_dir = tmp_path / "bad-task"
        task_dir.mkdir()
        (task_dir / "task.md").write_text(
            "---\nname: Bad Task\n---\n\n## Worker Goal\nGoal only\n"
        )
        with pytest.raises(ValueError, match="missing required sections"):
            load_task(task_dir)

    def test_defaults_name_from_dir(self, tmp_path: Path):
        task_dir = tmp_path / "folder-name"
        task_dir.mkdir()
        (task_dir / "task.md").write_text(
            "---\n---\n\n"
            "## Worker Goal\nGoal\n\n"
            "## Worker Instructions\nInstructions\n\n"
            "## Reviewer Criteria\nCriteria\n"
        )
        task = load_task(task_dir)
        assert task.name == "folder-name"

    def test_optional_fields(self, tmp_path: Path):
        task_dir = tmp_path / "opt-task"
        task_dir.mkdir()
        (task_dir / "task.md").write_text(
            "---\nname: Opt Task\nworker_model: openai\noutput_filename: index.html\n---\n\n"
            "## Worker Goal\nGoal\n\n"
            "## Worker Instructions\nInstructions\n\n"
            "## Reviewer Criteria\nCriteria\n"
        )
        task = load_task(task_dir)
        assert task.worker_model == "openai"
        assert task.output_filename == "index.html"

    def test_default_output_filename(self, tmp_path: Path):
        task_dir = tmp_path / "default-task"
        task_dir.mkdir()
        (task_dir / "task.md").write_text(
            "---\nname: Default Task\n---\n\n"
            "## Worker Goal\nGoal\n\n"
            "## Worker Instructions\nInstructions\n\n"
            "## Reviewer Criteria\nCriteria\n"
        )
        task = load_task(task_dir)
        assert task.output_filename == "output.md"


class TestLoadAllTasks:
    def test_load_multiple(self, tmp_path: Path):
        for name in ["task-a", "task-b"]:
            d = tmp_path / name
            d.mkdir()
            (d / "task.md").write_text(
                f"---\nname: {name}\n---\n\n"
                "## Worker Goal\nGoal\n\n"
                "## Worker Instructions\nInstructions\n\n"
                "## Reviewer Criteria\nCriteria\n"
            )
        tasks = load_all_tasks(tmp_path)
        assert len(tasks) == 2
        assert tasks[0].name == "task-a"
        assert tasks[1].name == "task-b"

    def test_empty_dir(self, tmp_path: Path):
        assert load_all_tasks(tmp_path) == []

    def test_nonexistent_dir(self, tmp_path: Path):
        assert load_all_tasks(tmp_path / "nope") == []

    def test_skips_dirs_without_task_md(self, tmp_path: Path):
        (tmp_path / "no-task").mkdir()
        assert load_all_tasks(tmp_path) == []
