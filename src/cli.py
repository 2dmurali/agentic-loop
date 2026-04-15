from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import click

from src.config import load_config
from src.runner import run_all_tasks
from src.task_loader import load_all_tasks


def _setup_logging(level: str, log_file: str | None = None) -> None:
    """Configure logging for the application."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


@click.group()
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), default=None, help="Path to config.yaml")
@click.pass_context
def main(ctx: click.Context, config_path: Path | None) -> None:
    """Agentic Loop — Worker-Reviewer iterative task runner."""
    ctx.ensure_object(dict)
    config = load_config(config_path)
    ctx.obj["config"] = config
    _setup_logging(config.logging.get("level", "INFO"), config.logging.get("file"))


@main.command()
@click.option("--task", "task_name", default=None, help="Run a specific task by folder name")
@click.option("--parallel", is_flag=True, default=False, help="Run tasks in parallel")
@click.option("--resume", is_flag=True, default=False, help="Resume the latest run instead of starting fresh")
@click.option("--tasks-dir", type=click.Path(exists=True, path_type=Path), default=None, help="Path to tasks directory")
@click.pass_context
def run(ctx: click.Context, task_name: str | None, parallel: bool, resume: bool, tasks_dir: Path | None) -> None:
    """Run tasks through the worker-reviewer loop."""
    config = ctx.obj["config"]
    if parallel:
        config.execution_mode = "parallel"

    results = asyncio.run(run_all_tasks(config, tasks_dir, task_name, parallel, resume))

    click.echo("\n--- Results ---")
    for name, result in results.items():
        status_icon = "✓" if result.status == "completed" else "⚠"
        score = f"{result.final_score}/10" if result.final_score is not None else "N/A"
        click.echo(f"{status_icon} {name}: {result.status} ({result.iterations} iterations, score: {score})")


@main.command()
@click.option("--tasks-dir", type=click.Path(exists=True, path_type=Path), default=None, help="Path to tasks directory")
def validate(tasks_dir: Path | None) -> None:
    """Validate task definitions without running LLM calls."""
    tasks = load_all_tasks(tasks_dir)
    if not tasks:
        click.echo("No tasks found.")
        return

    click.echo(f"Found {len(tasks)} task(s):\n")
    for task in tasks:
        click.echo(f"  ✓ {task.name}")
        click.echo(f"    min_reviews: {task.min_reviews}, max_reviews: {task.max_reviews}")
        click.echo(f"    worker_model: {task.worker_model or '(default)'}")
        click.echo(f"    reviewer_model: {task.reviewer_model or '(default)'}")
        click.echo(f"    output_filename: {task.output_filename}")
        click.echo()

    click.echo("All tasks validated successfully.")


@main.command()
@click.option("--tasks-dir", type=click.Path(exists=True, path_type=Path), default=None, help="Path to tasks directory")
def status(tasks_dir: Path | None) -> None:
    """Show the status of all tasks."""
    tasks = load_all_tasks(tasks_dir)
    if not tasks:
        click.echo("No tasks found.")
        return

    click.echo(f"{'Task':<30} {'Status':<20} {'Iterations':<12} {'Runs':<6}")
    click.echo("-" * 78)

    for task in tasks:
        output_dir = task.task_dir / "output"
        run_dirs = sorted(output_dir.glob("run-*")) if output_dir.exists() else []
        num_runs = len(run_dirs)

        # Check latest run
        latest_link = output_dir / "latest"
        if latest_link.is_symlink():
            latest_dir = latest_link.resolve()
            summary_path = latest_dir / "summary.md"
            if summary_path.exists():
                content = summary_path.read_text()
                if "Completed" in content:
                    task_status = "completed"
                elif "Max Iterations" in content:
                    task_status = "max iterations"
                else:
                    task_status = "done"
            else:
                task_status = "in progress"
            iterations = len(list(latest_dir.glob("worker-v*.md")))
        elif num_runs > 0:
            task_status = "has runs"
            iterations = len(list(run_dirs[-1].glob("worker-v*.md")))
        else:
            task_status = "pending"
            iterations = 0

        click.echo(f"{task.name:<30} {task_status:<20} {iterations:<12} {num_runs:<6}")


if __name__ == "__main__":
    main()
