from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from src.config import Config, ModelConfig
from src.llm.factory import create_provider
from src.llm.claude_provider import ClaudeProvider
from src.llm.openai_provider import OpenAIProvider


def _make_config(provider: str = "claude") -> Config:
    return Config(
        default_model="test",
        models={
            "test": ModelConfig(
                provider=provider,
                api_key_env="TEST_API_KEY",
                model_id="test-model",
                max_tokens=1024,
            )
        },
    )


class TestFactory:
    def test_create_claude_provider(self):
        config = _make_config("claude")
        with patch.dict(os.environ, {"TEST_API_KEY": "sk-test"}):
            provider = create_provider("test", config)
        assert isinstance(provider, ClaudeProvider)

    def test_create_openai_provider(self):
        config = _make_config("openai")
        with patch.dict(os.environ, {"TEST_API_KEY": "sk-test"}):
            provider = create_provider("test", config)
        assert isinstance(provider, OpenAIProvider)

    def test_unknown_provider_raises(self):
        config = Config(
            default_model="bad",
            models={
                "bad": ModelConfig(
                    provider="unknown",
                    api_key_env="TEST_API_KEY",
                    model_id="m",
                )
            },
        )
        with patch.dict(os.environ, {"TEST_API_KEY": "sk-test"}):
            with pytest.raises(ValueError, match="Unknown provider"):
                create_provider("bad", config)

    def test_missing_api_key_raises(self):
        config = _make_config("claude")
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EnvironmentError, match="TEST_API_KEY"):
                create_provider("test", config)

    def test_unknown_model_raises(self):
        config = _make_config("claude")
        with pytest.raises(KeyError):
            create_provider("nonexistent", config)
