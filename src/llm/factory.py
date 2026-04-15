from __future__ import annotations

import os

from src.config import Config
from src.llm.base import LLMProvider
from src.llm.claude_provider import ClaudeProvider
from src.llm.openai_provider import OpenAIProvider


def create_provider(model_name: str, config: Config) -> LLMProvider:
    """Create an LLM provider instance based on model name and configuration.

    Looks up the model in config.models, reads the API key from the
    environment variable specified in the model config, and instantiates
    the appropriate provider class.

    Raises:
        KeyError: If model_name is not found in config.models.
        ValueError: If the provider type is unknown.
        EnvironmentError: If the required API key environment variable is not set.
    """
    model_config = config.models[model_name]

    api_key = ""
    if model_config.api_key_env:
        api_key = os.environ.get(model_config.api_key_env, "")
    if not api_key and model_config.provider in ("claude", "openai"):
        raise EnvironmentError(
            f"Environment variable {model_config.api_key_env!r} is not set"
        )

    if model_config.provider == "claude":
        return ClaudeProvider(
            api_key=api_key,
            model_id=model_config.model_id,
            max_tokens=model_config.max_tokens,
        )
    elif model_config.provider in ("openai", "ollama"):
        return OpenAIProvider(
            api_key=api_key or "ollama",
            model_id=model_config.model_id,
            max_tokens=model_config.max_tokens,
            base_url=model_config.base_url,
        )
    else:
        raise ValueError(f"Unknown provider: {model_config.provider!r}")
