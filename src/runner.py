from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

from src.config import Config, load_config, resolve_task_config
from src.context import ContextAccumulator
from src.llm.factory import create_provider
from src.models import Message, ReviewResult, Task, TaskResult
from src.task_loader import load_all_tasks, load_task

logger = logging.getLogger("agentic-loop")


def _extract_json_from_text(text: str) -> dict | None:
    """Try to find and parse a JSON block embedded in free-text output."""
    import re
    # Try ```json ... ``` fenced blocks first
    match = re.search(r"```json\s*\n(.*?)\n\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try any {...} block that looks like our review schema
    for match in re.finditer(r"\{[^{}]*\"verdict\"[^{}]*\}", text, re.DOTALL):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    return None


def _infer_verdict_from_text(text: str) -> tuple[str, int]:
    """Heuristic: infer verdict and score from free-text reviewer output."""
    import re
    text_lower = text.lower()

    # Try to find a numeric score
    score = 0
    score_patterns = [
        r"(?:overall|final|total)\s*(?:score|rating)[:\s]*(\d+(?:\.\d+)?)\s*/\s*10",
        r"(\d+(?:\.\d+)?)\s*/\s*10",
        r"(?:score|rating)[:\s]*(\d+(?:\.\d+)?)",
    ]
    for pattern in score_patterns:
        m = re.search(pattern, text_lower)
        if m:
            score = min(int(float(m.group(1))), 10)
            break

    # Fallback: map letter grades to numeric scores
    if score == 0:
        grade_map = {
            "a+": 10, "a": 9, "a-": 8,
            "b+": 7, "b": 6, "b-": 5,
            "c+": 4, "c": 3, "c-": 2,
            "d": 1, "f": 0,
        }
        grade_pattern = r"(?:grade|rating|overall)\s*(?:\*{0,2})\s*[:\s]\s*(?:\*{0,2})\s*([a-f][+-]?)"
        m = re.search(grade_pattern, text_lower)
        if m:
            score = grade_map.get(m.group(1), 0)

    # Infer verdict from language
    approve_signals = ["approve", "accepted", "passes", "ready for publication", "a+", "distinction"]
    revise_signals = ["revise", "revision needed", "needs revision", "reject", "needs work", "not ready", "requires revision", "significant revision", "needs improvement", "not yet ready"]

    verdict = "REVISE"
    for signal in approve_signals:
        if signal in text_lower:
            verdict = "APPROVE"
            break
    for signal in revise_signals:
        if signal in text_lower:
            verdict = "REVISE"
            break

    return verdict, score


def _parse_review(raw: str) -> ReviewResult:
    """Parse structured JSON review from the reviewer's response.

    Tries three strategies:
    1. Parse the entire response as JSON
    2. Extract a JSON block from mixed text
    3. Infer verdict and score heuristically from free text
    """
    # Strategy 1: pure JSON
    try:
        data = json.loads(raw.strip())
        return ReviewResult(**data)
    except (json.JSONDecodeError, Exception):
        pass

    # Strategy 2: JSON embedded in text
    extracted = _extract_json_from_text(raw)
    if extracted:
        try:
            return ReviewResult(**extracted)
        except Exception:
            pass

    # Strategy 3: heuristic extraction from free text
    verdict, score = _infer_verdict_from_text(raw)
    return ReviewResult(
        verdict=verdict,
        score=score,
        comments=[raw.strip()],
        action_items=[],
    )


def _create_run_dir(task: Task) -> Path:
    """Create a timestamped run directory and update the 'latest' symlink."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = task.task_dir / "output" / f"run-{timestamp}"
    # Add numeric suffix if timestamp collides
    if run_dir.exists():
        counter = 2
        while (task.task_dir / "output" / f"run-{timestamp}-{counter}").exists():
            counter += 1
        run_dir = task.task_dir / "output" / f"run-{timestamp}-{counter}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Update 'latest' symlink
    latest_link = task.task_dir / "output" / "latest"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(run_dir.name)

    return run_dir


def _save_artifact(run_dir: Path, filename: str, content: str) -> Path:
    """Save an artifact to the run directory."""
    path = run_dir / filename
    path.write_text(content)
    return path


def _save_output(run_dir: Path, filename: str, content: str) -> Path:
    """Save the deliverable output to run_dir/output/{filename}."""
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_text(content)
    return path


def _read_output(run_dir: Path, filename: str) -> str | None:
    """Read the current deliverable from run_dir/output/{filename}."""
    path = run_dir / "output" / filename
    if path.exists():
        return path.read_text()
    return None


def _generate_summary(task: Task, result: TaskResult, reviews: list[ReviewResult]) -> str:
    """Generate a summary.md for a completed task."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# Task Summary: {task.name}",
        "",
        f"- **Status:** {result.status.replace('_', ' ').title()}",
        f"- **Iterations:** {result.iterations} of {task.max_reviews} max",
        f"- **Final Score:** {result.final_score}/10" if result.final_score is not None else "- **Final Score:** N/A",
        f"- **Completed:** {now}",
        "",
        "## Review History",
        "| Iteration | Score | Verdict |",
        "|-----------|-------|---------|",
    ]
    for i, review in enumerate(reviews, 1):
        lines.append(f"| {i} | {review.score}/10 | {review.verdict} |")

    if reviews:
        last = reviews[-1]
        if last.comments:
            lines.append("")
            lines.append("## Final Review Comments")
            for c in last.comments:
                lines.append(f"- {c}")

    return "\n".join(lines) + "\n"


async def _call_llm_with_retry(provider, messages, system_prompt, max_retries: int = 3) -> str:
    """Call LLM with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            return await provider.generate(messages, system_prompt)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}. Retrying in {wait}s...")
            await asyncio.sleep(wait)
    raise RuntimeError("Unreachable")


def _detect_resume_iteration(run_dir: Path) -> int:
    """Detect the last completed iteration by checking existing output files."""
    if not run_dir.exists():
        return 0
    iteration = 0
    while (run_dir / f"review-v{iteration + 1}.md").exists():
        iteration += 1
    return iteration


def _get_latest_run_dir(task: Task) -> Path | None:
    """Get the most recent run directory if it exists."""
    latest_link = task.task_dir / "output" / "latest"
    if latest_link.is_symlink():
        return latest_link.resolve()
    return None


async def run_task(task: Task, config: Config, resume: bool = False) -> TaskResult:
    """Run the worker-reviewer loop for a single task."""
    logger.info(f"Starting task: {task.name}")

    worker_model_name, reviewer_model_name = resolve_task_config(task, config)
    worker_provider = create_provider(worker_model_name, config)
    reviewer_provider = create_provider(reviewer_model_name, config)

    # Check for resumability against the latest run
    if resume:
        run_dir = _get_latest_run_dir(task)
        if run_dir is None:
            logger.info(f"No previous run to resume for '{task.name}', starting fresh")
            run_dir = _create_run_dir(task)
    else:
        run_dir = _create_run_dir(task)

    context_path = run_dir / "context.json"
    resume_iteration = _detect_resume_iteration(run_dir)

    if resume_iteration > 0 and context_path.exists():
        logger.info(f"Resuming task '{task.name}' from iteration {resume_iteration + 1} in {run_dir.name}")
        context = ContextAccumulator.load(context_path, task)
        start_iteration = resume_iteration + 1
    else:
        context = ContextAccumulator(task)
        start_iteration = 1

    logger.info(f"[{task.name}] Run directory: {run_dir.name}")
    logger.info(f"[{task.name}] Output filename: {task.output_filename}")
    reviews: list[ReviewResult] = []

    for iteration in range(start_iteration, task.max_reviews + 1):
        logger.info(f"[{task.name}] Iteration {iteration}/{task.max_reviews}")

        # Inject current output for revision (iteration 2+)
        if iteration > 1:
            current = _read_output(run_dir, task.output_filename)
            if current:
                context.inject_current_output(current, iteration)

        # Worker phase
        logger.info(f"[{task.name}] Worker phase (iteration {iteration})")
        worker_output = await _call_llm_with_retry(
            worker_provider,
            context.get_worker_messages(),
            context.worker_system_prompt,
        )
        context.add_worker_output(iteration, worker_output)
        _save_artifact(run_dir, f"worker-v{iteration}.md", worker_output)
        _save_output(run_dir, task.output_filename, worker_output)
        logger.info(f"[{task.name}] Saved output: {task.output_filename}")

        # Reviewer phase
        logger.info(f"[{task.name}] Reviewer phase (iteration {iteration})")
        review_raw = await _call_llm_with_retry(
            reviewer_provider,
            context.get_reviewer_messages(),
            context.reviewer_system_prompt,
        )
        parsed_review = _parse_review(review_raw)
        context.add_review(iteration, parsed_review)
        reviews.append(parsed_review)
        _save_artifact(run_dir, f"review-v{iteration}.md", review_raw)

        # Save context for resumability
        context.save(context_path)

        logger.info(
            f"[{task.name}] Iteration {iteration} — "
            f"Verdict: {parsed_review.verdict}, Score: {parsed_review.score}/10"
        )

        # Check completion (only approve if min_reviews met and score >= threshold)
        meets_threshold = (
            task.pass_threshold is None or parsed_review.score >= task.pass_threshold
        )
        if parsed_review.verdict == "APPROVE" and iteration >= task.min_reviews and meets_threshold:
            result = TaskResult(
                status="completed",
                iterations=iteration,
                final_score=parsed_review.score,
            )
            summary = _generate_summary(task, result, reviews)
            _save_artifact(run_dir, "summary.md", summary)
            logger.info(f"[{task.name}] Completed after {iteration} iteration(s)")
            return result

    # Max reviews reached
    result = TaskResult(
        status="max_iterations_reached",
        iterations=task.max_reviews,
        final_score=reviews[-1].score if reviews else None,
    )
    summary = _generate_summary(task, result, reviews)
    _save_artifact(run_dir, "summary.md", summary)
    logger.warning(f"[{task.name}] Max iterations reached ({task.max_reviews})")
    return result


async def run_all_tasks(
    config: Config,
    tasks_dir: Path | None = None,
    task_name: str | None = None,
    parallel: bool = False,
    resume: bool = False,
) -> dict[str, TaskResult]:
    """Run all tasks (or a specific one) and return results."""
    if task_name:
        base_dir = tasks_dir or Path.cwd() / "tasks"
        task_dir = base_dir / task_name
        if not task_dir.exists():
            raise FileNotFoundError(f"Task not found: {task_dir}")
        tasks = [load_task(task_dir)]
    else:
        tasks = load_all_tasks(tasks_dir)

    if not tasks:
        logger.warning("No tasks found")
        return {}

    results: dict[str, TaskResult] = {}

    if parallel:
        sem = asyncio.Semaphore(config.max_parallel_tasks)

        async def _run_with_sem(t: Task) -> tuple[str, TaskResult]:
            async with sem:
                return t.name, await run_task(t, config, resume)

        completed = await asyncio.gather(*[_run_with_sem(t) for t in tasks])
        results = dict(completed)
    else:
        for task in tasks:
            results[task.name] = await run_task(task, config, resume)

    return results
