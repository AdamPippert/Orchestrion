"""
Model information database for routing and cost tracking.

Contains known capabilities, pricing, and performance characteristics
for popular models across providers.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
from .base import ModelCapabilities, ProviderType


@dataclass
class ModelInfo:
    """Complete information about a model."""
    name: str
    provider: ProviderType
    capabilities: ModelCapabilities
    display_name: Optional[str] = None
    description: Optional[str] = None
    deprecated: bool = False

    # Aliases this model is known by
    aliases: tuple = ()


# Model database
_MODEL_DATABASE: Dict[str, ModelInfo] = {}


def _register_model(info: ModelInfo) -> None:
    """Register a model in the database."""
    _MODEL_DATABASE[info.name] = info
    for alias in info.aliases:
        _MODEL_DATABASE[alias] = info


def get_model_info(model_name: str) -> Optional[ModelInfo]:
    """Look up model information by name or alias."""
    return _MODEL_DATABASE.get(model_name)


def list_models(provider: Optional[ProviderType] = None) -> list[ModelInfo]:
    """List all known models, optionally filtered by provider."""
    seen = set()
    models = []
    for info in _MODEL_DATABASE.values():
        if info.name not in seen:
            if provider is None or info.provider == provider:
                models.append(info)
                seen.add(info.name)
    return models


# =============================================================================
# OpenAI Models
# =============================================================================

_register_model(ModelInfo(
    name="gpt-4o",
    provider=ProviderType.OPENAI,
    display_name="GPT-4o",
    description="Most capable OpenAI model, multimodal",
    capabilities=ModelCapabilities(
        chat=True,
        vision=True,
        function_calling=True,
        structured_output=True,
        streaming=True,
        max_context_length=128000,
        max_output_tokens=16384,
        input_cost_per_million=2.50,
        output_cost_per_million=10.00,
    ),
    aliases=("gpt-4o-2024-11-20",),
))

_register_model(ModelInfo(
    name="gpt-4o-mini",
    provider=ProviderType.OPENAI,
    display_name="GPT-4o Mini",
    description="Fast and affordable GPT-4o variant",
    capabilities=ModelCapabilities(
        chat=True,
        vision=True,
        function_calling=True,
        structured_output=True,
        streaming=True,
        max_context_length=128000,
        max_output_tokens=16384,
        input_cost_per_million=0.15,
        output_cost_per_million=0.60,
    ),
))

_register_model(ModelInfo(
    name="o1",
    provider=ProviderType.OPENAI,
    display_name="o1",
    description="Advanced reasoning model",
    capabilities=ModelCapabilities(
        chat=True,
        vision=True,
        function_calling=True,
        structured_output=True,
        streaming=True,
        max_context_length=200000,
        max_output_tokens=100000,
        input_cost_per_million=15.00,
        output_cost_per_million=60.00,
    ),
))

_register_model(ModelInfo(
    name="o1-mini",
    provider=ProviderType.OPENAI,
    display_name="o1 Mini",
    description="Fast reasoning model",
    capabilities=ModelCapabilities(
        chat=True,
        function_calling=True,
        structured_output=True,
        streaming=True,
        max_context_length=128000,
        max_output_tokens=65536,
        input_cost_per_million=3.00,
        output_cost_per_million=12.00,
    ),
))

_register_model(ModelInfo(
    name="text-embedding-3-small",
    provider=ProviderType.OPENAI,
    display_name="Text Embedding 3 Small",
    description="Efficient embedding model",
    capabilities=ModelCapabilities(
        chat=False,
        embeddings=True,
        streaming=False,
        input_cost_per_million=0.02,
        output_cost_per_million=0.0,
    ),
))

_register_model(ModelInfo(
    name="text-embedding-3-large",
    provider=ProviderType.OPENAI,
    display_name="Text Embedding 3 Large",
    description="High-quality embedding model",
    capabilities=ModelCapabilities(
        chat=False,
        embeddings=True,
        streaming=False,
        input_cost_per_million=0.13,
        output_cost_per_million=0.0,
    ),
))

# =============================================================================
# Anthropic Models
# =============================================================================

_register_model(ModelInfo(
    name="claude-sonnet-4-20250514",
    provider=ProviderType.ANTHROPIC,
    display_name="Claude Sonnet 4",
    description="Latest Claude Sonnet - balanced performance",
    capabilities=ModelCapabilities(
        chat=True,
        vision=True,
        function_calling=True,
        structured_output=True,
        streaming=True,
        max_context_length=200000,
        max_output_tokens=64000,
        input_cost_per_million=3.00,
        output_cost_per_million=15.00,
    ),
    aliases=("claude-4-sonnet", "claude-sonnet-4"),
))

_register_model(ModelInfo(
    name="claude-opus-4-20250514",
    provider=ProviderType.ANTHROPIC,
    display_name="Claude Opus 4",
    description="Most capable Claude model",
    capabilities=ModelCapabilities(
        chat=True,
        vision=True,
        function_calling=True,
        structured_output=True,
        streaming=True,
        max_context_length=200000,
        max_output_tokens=64000,
        input_cost_per_million=15.00,
        output_cost_per_million=75.00,
    ),
    aliases=("claude-4-opus", "claude-opus-4"),
))

_register_model(ModelInfo(
    name="claude-3-5-sonnet-20241022",
    provider=ProviderType.ANTHROPIC,
    display_name="Claude 3.5 Sonnet",
    description="Previous generation Claude Sonnet",
    capabilities=ModelCapabilities(
        chat=True,
        vision=True,
        function_calling=True,
        structured_output=True,
        streaming=True,
        max_context_length=200000,
        max_output_tokens=8192,
        input_cost_per_million=3.00,
        output_cost_per_million=15.00,
    ),
    aliases=("claude-3.5-sonnet", "claude-3-5-sonnet"),
))

_register_model(ModelInfo(
    name="claude-3-5-haiku-20241022",
    provider=ProviderType.ANTHROPIC,
    display_name="Claude 3.5 Haiku",
    description="Fast and efficient Claude model",
    capabilities=ModelCapabilities(
        chat=True,
        vision=True,
        function_calling=True,
        structured_output=True,
        streaming=True,
        max_context_length=200000,
        max_output_tokens=8192,
        input_cost_per_million=0.80,
        output_cost_per_million=4.00,
    ),
    aliases=("claude-3.5-haiku", "claude-3-5-haiku"),
))

# =============================================================================
# Local / Open Models (typical configurations)
# =============================================================================

_register_model(ModelInfo(
    name="llama3.2",
    provider=ProviderType.OLLAMA,
    display_name="Llama 3.2",
    description="Meta's Llama 3.2 (via Ollama)",
    capabilities=ModelCapabilities(
        chat=True,
        vision=False,
        function_calling=True,
        streaming=True,
        max_context_length=128000,
        max_output_tokens=4096,
        input_cost_per_million=0.0,  # Local = free
        output_cost_per_million=0.0,
    ),
    aliases=("llama-3.2", "llama3.2:latest"),
))

_register_model(ModelInfo(
    name="llama3.2-vision",
    provider=ProviderType.OLLAMA,
    display_name="Llama 3.2 Vision",
    description="Meta's Llama 3.2 with vision (via Ollama)",
    capabilities=ModelCapabilities(
        chat=True,
        vision=True,
        function_calling=True,
        streaming=True,
        max_context_length=128000,
        max_output_tokens=4096,
        input_cost_per_million=0.0,
        output_cost_per_million=0.0,
    ),
))

_register_model(ModelInfo(
    name="mistral",
    provider=ProviderType.OLLAMA,
    display_name="Mistral",
    description="Mistral 7B (via Ollama)",
    capabilities=ModelCapabilities(
        chat=True,
        function_calling=True,
        streaming=True,
        max_context_length=32768,
        max_output_tokens=4096,
        input_cost_per_million=0.0,
        output_cost_per_million=0.0,
    ),
))

_register_model(ModelInfo(
    name="qwen2.5",
    provider=ProviderType.OLLAMA,
    display_name="Qwen 2.5",
    description="Alibaba's Qwen 2.5 (via Ollama)",
    capabilities=ModelCapabilities(
        chat=True,
        function_calling=True,
        streaming=True,
        max_context_length=32768,
        max_output_tokens=4096,
        input_cost_per_million=0.0,
        output_cost_per_million=0.0,
    ),
))

_register_model(ModelInfo(
    name="deepseek-r1",
    provider=ProviderType.VLLM,
    display_name="DeepSeek R1",
    description="DeepSeek R1 reasoning model",
    capabilities=ModelCapabilities(
        chat=True,
        function_calling=True,
        streaming=True,
        max_context_length=64000,
        max_output_tokens=8192,
        input_cost_per_million=0.0,  # Local
        output_cost_per_million=0.0,
    ),
))
