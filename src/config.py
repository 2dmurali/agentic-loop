from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml
from pydantic import BaseModel

if TYPE_CHECKING:
    from src.models import Task


class ModelConfig(BaseModel):
    provider: str
    api_key_env: str = ""
    model_id: str
    max_tokens: int = 4096
    base_url: str | None = None


class Config(BaseModel):
    default_model: str = "claude"
    execution_mode: Literal["sequential", "parallel"] = "sequential"
    max_parallel_tasks: int = 3
    models: dict[str, ModelConfig]
    defaults: dict = {"min_reviews": 1, "max_reviews": 5}
    logging: dict = {"level": "INFO", "file": "agentic-loop.log"}


def load_config(config_path: Path | None = None) -> Config:
    """Load configuration from a YAML file and return a Config instance.

    If config_path is None, looks for config.yaml in the current working directory.
    Raises FileNotFoundError if the configuration file does not exist.
    """
    if config_path is None:
        config_path = Path.cwd() / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    return Config(**raw)


def resolve_task_config(task: Task, config: Config) -> tuple[str, str]:
    """Resolve the worker and reviewer model names for a given task.

    Returns a tuple of (worker_model_name, reviewer_model_name).
    Uses task-level overrides if specified, otherwise falls back to
    config.default_model.
    """
    worker_model = task.worker_model if task.worker_model else config.default_model
    reviewer_model = task.reviewer_model if task.reviewer_model else config.default_model
    return worker_model, reviewer_model
