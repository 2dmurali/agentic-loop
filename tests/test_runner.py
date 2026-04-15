from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.config import Config, ModelConfig
from src.models import Task
from src.runner import (
    _parse_review,
    _generate_summary,
    _save_artifact,
    _save_output,
    _read_output,
    _create_run_dir,
    run_task,
)


def _make_config() -> Config:
    return Config(
        default_model="mock",
        models={
            "mock": ModelConfig(
                provider="claude",
                api_key_env="MOCK_KEY",
                model_id="mock-model",
            )
        },
    )


def _make_task(task_dir: Path, **overrides) -> Task:
    defaults = dict(
        name="test-task",
        min_reviews=1,
        max_reviews=3,
        worker_goal="Write a hello world",
        worker_instructions="Use Python",
        reviewer_criteria="Check correctness",
        task_dir=task_dir,
    )
    defaults.update(overrides)
    return Task(**defaults)


def _get_latest_run(task_dir: Path) -> Path:
    """Resolve the latest run directory."""
    latest = task_dir / "output" / "latest"
    return latest.resolve()


class TestParseReview:
    def test_valid_json(self):
        raw = json.dumps({
            "verdict": "APPROVE",
            "score": 8,
            "comments": ["Good work"],
            "action_items": [],
        })
        result = _parse_review(raw)
        assert result.verdict == "APPROVE"
        assert result.score == 8

    def test_revise_verdict(self):
        raw = json.dumps({
            "verdict": "REVISE",
            "score": 3,
            "comments": ["Needs work"],
            "action_items": ["Fix bug"],
        })
        result = _parse_review(raw)
        assert result.verdict == "REVISE"
        assert len(result.action_items) == 1

    def test_invalid_json_fallback(self):
        result = _parse_review("This is not JSON at all")
        assert result.verdict == "REVISE"
        assert result.score == 0
        assert "This is not JSON at all" in result.comments[0]

    def test_empty_string(self):
        result = _parse_review("")
        assert result.verdict == "REVISE"

    def test_json_embedded_in_text(self):
        raw = 'Here is my review:\n```json\n{"verdict": "APPROVE", "score": 9, "comments": ["Great"], "action_items": []}\n```\nThat is all.'
        result = _parse_review(raw)
        assert result.verdict == "APPROVE"
        assert result.score == 9

    def test_heuristic_score_extraction(self):
        raw = "Good work overall. Final score: 8/10. Needs minor revision."
        result = _parse_review(raw)
        assert result.score == 8

    def test_heuristic_approve_detection(self):
        raw = "Excellent work. Overall score: 9/10. Approved for publication."
        result = _parse_review(raw)
        assert result.verdict == "APPROVE"
        assert result.score == 9


class TestSaveArtifact:
    def test_saves_to_run_dir(self, tmp_path: Path):
        run_dir = tmp_path / "run-test"
        run_dir.mkdir()
        path = _save_artifact(run_dir, "worker-v1.md", "Hello")
        assert path.exists()
        assert path.read_text() == "Hello"


class TestSaveAndReadOutput:
    def test_save_creates_output_dir(self, tmp_path: Path):
        run_dir = tmp_path / "run-test"
        run_dir.mkdir()
        path = _save_output(run_dir, "index.html", "<html>Hello</html>")
        assert path.exists()
        assert path.read_text() == "<html>Hello</html>"
        assert path.parent.name == "output"

    def test_read_existing_output(self, tmp_path: Path):
        run_dir = tmp_path / "run-test"
        (run_dir / "output").mkdir(parents=True)
        (run_dir / "output" / "main.py").write_text("print('hi')")
        content = _read_output(run_dir, "main.py")
        assert content == "print('hi')"

    def test_read_nonexistent_output(self, tmp_path: Path):
        run_dir = tmp_path / "run-test"
        run_dir.mkdir()
        assert _read_output(run_dir, "missing.py") is None

    def test_save_overwrites_existing(self, tmp_path: Path):
        run_dir = tmp_path / "run-test"
        run_dir.mkdir()
        _save_output(run_dir, "file.py", "v1")
        _save_output(run_dir, "file.py", "v2")
        assert _read_output(run_dir, "file.py") == "v2"


class TestCreateRunDir:
    def test_creates_timestamped_dir(self, tmp_path: Path):
        task = _make_task(tmp_path)
        run_dir = _create_run_dir(task)
        assert run_dir.exists()
        assert run_dir.name.startswith("run-")
        assert (tmp_path / "output" / "latest").is_symlink()

    def test_latest_points_to_newest(self, tmp_path: Path):
        task = _make_task(tmp_path)
        run1 = _create_run_dir(task)
        run2 = _create_run_dir(task)
        latest = (tmp_path / "output" / "latest").resolve()
        assert latest == run2


class TestRunTask:
    @pytest.mark.asyncio
    async def test_approve_on_first_iteration(self, tmp_path: Path):
        task = _make_task(tmp_path, min_reviews=1, max_reviews=3)
        config = _make_config()

        approve_review = json.dumps({
            "verdict": "APPROVE",
            "score": 9,
            "comments": ["Perfect"],
            "action_items": [],
        })

        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(side_effect=["Worker output v1", approve_review])

        with patch("src.runner.create_provider", return_value=mock_provider):
            result = await run_task(task, config)

        assert result.status == "completed"
        assert result.iterations == 1
        assert result.final_score == 9
        run_dir = _get_latest_run(tmp_path)
        assert (run_dir / "worker-v1.md").exists()
        assert (run_dir / "review-v1.md").exists()
        assert (run_dir / "summary.md").exists()
        # Output deliverable should exist
        assert (run_dir / "output" / "output.md").exists()
        assert (run_dir / "output" / "output.md").read_text() == "Worker output v1"

    @pytest.mark.asyncio
    async def test_code_task_saves_output_file(self, tmp_path: Path):
        task = _make_task(tmp_path, min_reviews=1, max_reviews=3, output_filename="index.html")
        config = _make_config()

        html_content = "<!DOCTYPE html><html><body>Game</body></html>"
        approve_review = json.dumps({
            "verdict": "APPROVE",
            "score": 9,
            "comments": ["Great game"],
            "action_items": [],
        })

        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(side_effect=[html_content, approve_review])

        with patch("src.runner.create_provider", return_value=mock_provider):
            result = await run_task(task, config)

        assert result.status == "completed"
        run_dir = _get_latest_run(tmp_path)
        assert (run_dir / "output" / "index.html").exists()
        assert (run_dir / "output" / "index.html").read_text() == html_content

    @pytest.mark.asyncio
    async def test_min_reviews_enforced(self, tmp_path: Path):
        task = _make_task(tmp_path, min_reviews=2, max_reviews=3)
        config = _make_config()

        responses = [
            "Worker output v1",
            json.dumps({"verdict": "APPROVE", "score": 8, "comments": ["Good"], "action_items": []}),
            "Worker output v2",
            json.dumps({"verdict": "APPROVE", "score": 9, "comments": ["Great"], "action_items": []}),
        ]

        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(side_effect=responses)

        with patch("src.runner.create_provider", return_value=mock_provider):
            result = await run_task(task, config)

        assert result.status == "completed"
        assert result.iterations == 2

    @pytest.mark.asyncio
    async def test_max_reviews_reached(self, tmp_path: Path):
        task = _make_task(tmp_path, min_reviews=1, max_reviews=2)
        config = _make_config()

        responses = [
            "Worker v1",
            json.dumps({"verdict": "REVISE", "score": 3, "comments": ["Bad"], "action_items": ["Fix"]}),
            "Worker v2",
            json.dumps({"verdict": "REVISE", "score": 5, "comments": ["Better"], "action_items": ["Polish"]}),
        ]

        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(side_effect=responses)

        with patch("src.runner.create_provider", return_value=mock_provider):
            result = await run_task(task, config)

        assert result.status == "max_iterations_reached"
        assert result.iterations == 2
        run_dir = _get_latest_run(tmp_path)
        assert (run_dir / "worker-v1.md").exists()
        assert (run_dir / "worker-v2.md").exists()
        assert (run_dir / "summary.md").exists()
        # Output should have the latest version
        assert (run_dir / "output" / "output.md").read_text() == "Worker v2"

    @pytest.mark.asyncio
    async def test_context_persisted(self, tmp_path: Path):
        task = _make_task(tmp_path, min_reviews=1, max_reviews=2)
        config = _make_config()

        responses = [
            "Worker v1",
            json.dumps({"verdict": "REVISE", "score": 4, "comments": ["Improve"], "action_items": []}),
            "Worker v2",
            json.dumps({"verdict": "APPROVE", "score": 9, "comments": ["Done"], "action_items": []}),
        ]

        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(side_effect=responses)

        with patch("src.runner.create_provider", return_value=mock_provider):
            await run_task(task, config)

        run_dir = _get_latest_run(tmp_path)
        assert (run_dir / "context.json").exists()

    @pytest.mark.asyncio
    async def test_multiple_runs_preserved(self, tmp_path: Path):
        task = _make_task(tmp_path, min_reviews=1, max_reviews=1)
        config = _make_config()

        approve = json.dumps({"verdict": "APPROVE", "score": 9, "comments": ["OK"], "action_items": []})

        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(side_effect=["Work 1", approve, "Work 2", approve])

        with patch("src.runner.create_provider", return_value=mock_provider):
            await run_task(task, config)
            await run_task(task, config)

        # Both runs should exist
        output_dir = tmp_path / "output"
        run_dirs = list(output_dir.glob("run-*"))
        assert len(run_dirs) == 2

    @pytest.mark.asyncio
    async def test_pass_threshold_blocks_low_score(self, tmp_path: Path):
        task = _make_task(tmp_path, min_reviews=1, max_reviews=2, pass_threshold=9.0)
        config = _make_config()

        responses = [
            "Worker v1",
            json.dumps({"verdict": "APPROVE", "score": 7, "comments": ["Good but not great"], "action_items": []}),
            "Worker v2",
            json.dumps({"verdict": "APPROVE", "score": 9, "comments": ["Excellent"], "action_items": []}),
        ]

        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(side_effect=responses)

        with patch("src.runner.create_provider", return_value=mock_provider):
            result = await run_task(task, config)

        assert result.status == "completed"
        assert result.iterations == 2  # First approve blocked by threshold
        assert result.final_score == 9
