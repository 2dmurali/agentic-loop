from __future__ import annotations

import re
from pathlib import Path

import frontmatter

from src.models import Task


def _parse_sections(content: str) -> dict[str, str]:
    """Split markdown content by ## headings.

    Returns a dict mapping heading text to the section body (stripped).
    """
    sections: dict[str, str] = {}
    # Split on lines that start with '## '
    parts = re.split(r"^## ", content, flags=re.MULTILINE)
    for part in parts[1:]:  # skip anything before the first ## heading
        lines = part.split("\n", 1)
        heading = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        sections[heading] = body
    return sections


def load_task(task_dir: Path) -> Task:
    """Load a single task from a directory containing a task.md file.

    Reads ``task_dir/task.md``, parses YAML frontmatter for configuration
    fields, and extracts markdown sections for worker/reviewer instructions.

    Raises:
        FileNotFoundError: If task.md does not exist in *task_dir*.
        ValueError: If any required section is missing from the markdown body.
    """
    task_file = task_dir / "task.md"
    post = frontmatter.load(str(task_file))

    # Extract frontmatter fields
    meta = post.metadata
    name: str = meta.get("name", task_dir.name)
    min_reviews: int = meta.get("min_reviews", 1)
    max_reviews: int = meta.get("max_reviews", 5)
    pass_threshold: float | None = meta.get("pass_threshold")
    worker_model: str | None = meta.get("worker_model")
    reviewer_model: str | None = meta.get("reviewer_model")
    output_filename: str = meta.get("output_filename", "output.md")

    # Parse markdown body into sections
    sections = _parse_sections(post.content)

    required = {
        "Worker Goal": "worker_goal",
        "Worker Instructions": "worker_instructions",
        "Reviewer Criteria": "reviewer_criteria",
    }

    missing = [heading for heading in required if heading not in sections]
    if missing:
        raise ValueError(
            f"task.md in {task_dir} is missing required sections: "
            + ", ".join(f'"## {h}"' for h in missing)
        )

    return Task(
        name=name,
        min_reviews=min_reviews,
        max_reviews=max_reviews,
        pass_threshold=pass_threshold,
        worker_model=worker_model,
        reviewer_model=reviewer_model,
        output_filename=output_filename,
        worker_goal=sections["Worker Goal"],
        worker_instructions=sections["Worker Instructions"],
        reviewer_criteria=sections["Reviewer Criteria"],
        task_dir=task_dir,
    )


def load_all_tasks(tasks_dir: Path | None = None) -> list[Task]:
    """Load all tasks from subdirectories of *tasks_dir*.

    Each subdirectory that contains a ``task.md`` file is treated as a task.
    Returns the loaded tasks sorted by name.  If *tasks_dir* does not exist,
    returns an empty list.
    """
    if tasks_dir is None:
        tasks_dir = Path.cwd() / "tasks"

    if not tasks_dir.exists():
        return []

    tasks: list[Task] = []
    for child in sorted(tasks_dir.iterdir()):
        if child.is_dir() and (child / "task.md").exists():
            tasks.append(load_task(child))

    tasks.sort(key=lambda t: t.name)
    return tasks
