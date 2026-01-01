"""
Provider abstraction layer for LLM and model inference.

Supports:
- Cloud providers: OpenAI, Anthropic, Google
- Local runtimes: vLLM, RamaLama, Ollama, llama.cpp
- Custom OpenAI-compatible endpoints

All providers implement a unified interface for easy swapping and routing.
"""

from .base import (
    BaseProvider,
    ProviderConfig,
    ProviderType,
    ModelCapabilities,
    ProviderRegistry,
    Message,
    CompletionResponse,
    StreamChunk,
)
from .openai_compat import OpenAICompatibleProvider
from .anthropic import AnthropicProvider
from .model_info import ModelInfo, get_model_info, list_models

__all__ = [
    'BaseProvider',
    'ProviderConfig',
    'ProviderType',
    'ModelCapabilities',
    'ProviderRegistry',
    'Message',
    'CompletionResponse',
    'StreamChunk',
    'OpenAICompatibleProvider',
    'AnthropicProvider',
    'ModelInfo',
    'get_model_info',
    'list_models',
]
