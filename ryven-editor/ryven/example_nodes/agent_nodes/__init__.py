"""
Orchestrion Agent Nodes

A comprehensive package for building AI agent workflows in Ryven/Orchestrion.

Features:
- LLM Provider Integration (OpenAI, Anthropic, vLLM, RamaLama, Ollama)
- Agent Definition (roles, personas, capabilities)
- Guardrails (validation, limits, approval gates, audit)
- Routing (cost/speed/quality optimization, semantic routing)
- Memory (vector stores, context management, knowledge bases)
- Observability (metrics, tracing, debugging)

Usage:
    Import this package in Ryven to access all agent-related nodes.
    See the README.md for detailed documentation.
"""

__version__ = "0.1.0"
__author__ = "Orchestrion Contributors"

# Core provider imports for programmatic use
from .providers import (
    BaseProvider,
    ProviderConfig,
    ProviderType,
    ProviderRegistry,
    OpenAICompatibleProvider,
    AnthropicProvider,
    Message,
    CompletionResponse,
)

from .providers.openai_compat import (
    create_openai_provider,
    create_vllm_provider,
    create_ramalama_provider,
    create_ollama_provider,
    create_llamacpp_provider,
)

from .providers.anthropic import create_anthropic_provider
from .providers.model_info import get_model_info, list_models, ModelInfo

# Agent definition exports
from .agents.definition import AgentDefinition

__all__ = [
    # Provider base
    'BaseProvider',
    'ProviderConfig',
    'ProviderType',
    'ProviderRegistry',
    'OpenAICompatibleProvider',
    'AnthropicProvider',
    'Message',
    'CompletionResponse',

    # Factory functions
    'create_openai_provider',
    'create_vllm_provider',
    'create_ramalama_provider',
    'create_ollama_provider',
    'create_llamacpp_provider',
    'create_anthropic_provider',

    # Model info
    'get_model_info',
    'list_models',
    'ModelInfo',

    # Agents
    'AgentDefinition',
]
