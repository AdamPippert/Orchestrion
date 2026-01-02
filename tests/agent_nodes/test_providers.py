"""
Tests for LLM provider abstraction layer.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add the package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../ryven-editor/ryven/example_nodes'))

from agent_nodes.providers.base import (
    BaseProvider,
    ProviderConfig,
    ProviderType,
    ProviderRegistry,
    Message,
    CompletionResponse,
    ModelCapabilities,
)
from agent_nodes.providers.model_info import get_model_info, list_models


class TestProviderConfig:
    """Tests for ProviderConfig."""

    def test_default_openai_url(self):
        config = ProviderConfig(provider_type=ProviderType.OPENAI)
        assert config.base_url == "https://api.openai.com/v1"

    def test_default_vllm_url(self):
        config = ProviderConfig(provider_type=ProviderType.VLLM)
        assert config.base_url == "http://localhost:8000/v1"

    def test_default_ollama_url(self):
        config = ProviderConfig(provider_type=ProviderType.OLLAMA)
        assert config.base_url == "http://localhost:11434/v1"

    def test_default_ramalama_url(self):
        config = ProviderConfig(provider_type=ProviderType.RAMALAMA)
        assert config.base_url == "http://localhost:8080/v1"

    def test_custom_url(self):
        config = ProviderConfig(
            provider_type=ProviderType.VLLM,
            base_url="http://custom:9000/v1"
        )
        assert config.base_url == "http://custom:9000/v1"

    def test_config_with_all_options(self):
        config = ProviderConfig(
            provider_type=ProviderType.OPENAI,
            api_key="test-key",
            model="gpt-4o",
            timeout=120.0,
            max_retries=5,
        )
        assert config.api_key == "test-key"
        assert config.model == "gpt-4o"
        assert config.timeout == 120.0
        assert config.max_retries == 5


class TestMessage:
    """Tests for Message class."""

    def test_basic_message(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_message_to_dict(self):
        msg = Message(role="user", content="Hello")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "Hello"}

    def test_message_with_name(self):
        msg = Message(role="user", content="Hello", name="Alice")
        d = msg.to_dict()
        assert d["name"] == "Alice"

    def test_message_with_tool_calls(self):
        tool_calls = [{"id": "1", "function": {"name": "test"}}]
        msg = Message(role="assistant", content="", tool_calls=tool_calls)
        d = msg.to_dict()
        assert d["tool_calls"] == tool_calls


class TestCompletionResponse:
    """Tests for CompletionResponse class."""

    def test_basic_response(self):
        resp = CompletionResponse(
            content="Hello!",
            model="gpt-4o",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
        assert resp.content == "Hello!"
        assert resp.model == "gpt-4o"
        assert resp.total_tokens == 15

    def test_response_with_tool_calls(self):
        resp = CompletionResponse(
            content="",
            model="gpt-4o",
            tool_calls=[{"id": "1", "function": {"name": "search"}}],
        )
        assert resp.tool_calls is not None
        assert len(resp.tool_calls) == 1


class TestProviderRegistry:
    """Tests for ProviderRegistry."""

    def setup_method(self):
        ProviderRegistry.clear()

    def test_register_provider(self):
        mock_provider = MagicMock(spec=BaseProvider)
        ProviderRegistry.register("test", mock_provider)
        assert ProviderRegistry.get("test") == mock_provider

    def test_set_default(self):
        mock_provider = MagicMock(spec=BaseProvider)
        ProviderRegistry.register("test", mock_provider, set_default=True)
        assert ProviderRegistry.get() == mock_provider

    def test_list_providers(self):
        mock1 = MagicMock(spec=BaseProvider)
        mock2 = MagicMock(spec=BaseProvider)
        ProviderRegistry.register("p1", mock1)
        ProviderRegistry.register("p2", mock2)
        names = ProviderRegistry.list_providers()
        assert "p1" in names
        assert "p2" in names

    def test_remove_provider(self):
        mock_provider = MagicMock(spec=BaseProvider)
        ProviderRegistry.register("test", mock_provider)
        ProviderRegistry.remove("test")
        with pytest.raises(ValueError):
            ProviderRegistry.get("test")

    def test_no_provider_error(self):
        with pytest.raises(ValueError):
            ProviderRegistry.get("nonexistent")


class TestModelInfo:
    """Tests for model information database."""

    def test_get_openai_model(self):
        info = get_model_info("gpt-4o")
        assert info is not None
        assert info.provider == ProviderType.OPENAI
        assert info.capabilities.chat is True
        assert info.capabilities.vision is True

    def test_get_anthropic_model(self):
        info = get_model_info("claude-sonnet-4-20250514")
        assert info is not None
        assert info.provider == ProviderType.ANTHROPIC

    def test_get_model_alias(self):
        info = get_model_info("claude-4-sonnet")  # Alias
        assert info is not None
        assert info.name == "claude-sonnet-4-20250514"

    def test_list_all_models(self):
        models = list_models()
        assert len(models) > 0

    def test_list_models_by_provider(self):
        openai_models = list_models(provider=ProviderType.OPENAI)
        for model in openai_models:
            assert model.provider == ProviderType.OPENAI

    def test_embedding_model_capabilities(self):
        info = get_model_info("text-embedding-3-small")
        assert info is not None
        assert info.capabilities.embeddings is True
        assert info.capabilities.chat is False


class TestModelCapabilities:
    """Tests for ModelCapabilities."""

    def test_default_capabilities(self):
        caps = ModelCapabilities()
        assert caps.chat is True
        assert caps.streaming is True
        assert caps.max_context_length == 4096

    def test_custom_capabilities(self):
        caps = ModelCapabilities(
            chat=True,
            vision=True,
            function_calling=True,
            max_context_length=128000,
            input_cost_per_million=2.50,
        )
        assert caps.vision is True
        assert caps.function_calling is True
        assert caps.max_context_length == 128000
        assert caps.input_cost_per_million == 2.50
