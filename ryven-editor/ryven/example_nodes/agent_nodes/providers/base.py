"""
Base provider abstraction for unified LLM/model access.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, AsyncIterator, Dict, List, Optional,
    Type, Union, Callable, TypeVar, Generic
)
import asyncio
import time
import logging

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Supported provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    VLLM = "vllm"
    RAMALAMA = "ramalama"
    OLLAMA = "ollama"
    LLAMACPP = "llamacpp"
    CUSTOM = "custom"


@dataclass
class ModelCapabilities:
    """Describes what a model can do."""
    chat: bool = True
    completion: bool = False
    embeddings: bool = False
    vision: bool = False
    audio: bool = False
    function_calling: bool = False
    structured_output: bool = False
    streaming: bool = True

    max_context_length: int = 4096
    max_output_tokens: int = 4096

    # Cost per 1M tokens (for routing decisions)
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0

    # Performance characteristics
    tokens_per_second: Optional[float] = None  # Typical generation speed
    latency_ms: Optional[float] = None  # Time to first token


@dataclass
class ProviderConfig:
    """Configuration for a provider instance."""
    provider_type: ProviderType
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = ""

    # Connection settings
    timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0

    # Rate limiting
    requests_per_minute: Optional[int] = None
    tokens_per_minute: Optional[int] = None

    # Additional provider-specific options
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Set default base URLs for known providers
        if self.base_url is None:
            defaults = {
                ProviderType.OPENAI: "https://api.openai.com/v1",
                ProviderType.ANTHROPIC: "https://api.anthropic.com",
                ProviderType.VLLM: "http://localhost:8000/v1",
                ProviderType.RAMALAMA: "http://localhost:8080/v1",
                ProviderType.OLLAMA: "http://localhost:11434/v1",
                ProviderType.LLAMACPP: "http://localhost:8080/v1",
            }
            self.base_url = defaults.get(self.provider_type)


@dataclass
class Message:
    """A chat message."""
    role: str  # "system", "user", "assistant", "tool"
    content: Union[str, List[Dict[str, Any]]]  # Text or multimodal content
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass
class CompletionResponse:
    """Unified response from any provider."""
    content: str
    model: str

    # Usage statistics
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Tool/function calls
    tool_calls: Optional[List[Dict[str, Any]]] = None

    # Structured output (if requested)
    parsed: Optional[Any] = None

    # Timing
    latency_ms: float = 0.0
    time_to_first_token_ms: Optional[float] = None

    # Provider metadata
    finish_reason: str = "stop"
    provider_type: Optional[ProviderType] = None
    raw_response: Optional[Dict[str, Any]] = None

    @property
    def cost(self) -> float:
        """Calculate cost based on token usage and model pricing."""
        # Will be populated by provider based on model info
        return 0.0


@dataclass
class StreamChunk:
    """A chunk from a streaming response."""
    content: str = ""
    finish_reason: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    # Running totals (if available)
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class BaseProvider(ABC):
    """
    Abstract base class for all LLM providers.

    Implementations should handle:
    - Authentication and connection management
    - Request/response transformation
    - Error handling and retries
    - Rate limiting
    - Streaming
    """

    def __init__(self, config: ProviderConfig):
        self.config = config
        self._client: Any = None
        self._last_request_time: float = 0
        self._request_count: int = 0

    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider client."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close connections and cleanup."""
        pass

    @abstractmethod
    async def chat(
        self,
        messages: List[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> CompletionResponse:
        """
        Send a chat completion request.

        Args:
            messages: List of conversation messages
            model: Model to use (defaults to config.model)
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            tools: List of tool definitions for function calling
            tool_choice: How to select tools ("auto", "none", or specific)
            response_format: JSON schema for structured output
            stop: Stop sequences
            **kwargs: Additional provider-specific parameters

        Returns:
            CompletionResponse with the model's response
        """
        pass

    @abstractmethod
    async def chat_stream(
        self,
        messages: List[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a chat completion response.

        Yields StreamChunk objects as they arrive.
        """
        pass

    async def embed(
        self,
        texts: List[str],
        *,
        model: Optional[str] = None,
        **kwargs,
    ) -> List[List[float]]:
        """
        Generate embeddings for texts.

        Override in providers that support embeddings.
        """
        raise NotImplementedError(
            f"{self.provider_type.value} does not support embeddings"
        )

    def get_capabilities(self, model: Optional[str] = None) -> ModelCapabilities:
        """Get capabilities for a specific model."""
        from .model_info import get_model_info
        model = model or self.config.model
        info = get_model_info(model)
        if info:
            return info.capabilities
        return ModelCapabilities()  # Default capabilities

    async def _rate_limit(self) -> None:
        """Apply rate limiting if configured."""
        if self.config.requests_per_minute:
            min_interval = 60.0 / self.config.requests_per_minute
            elapsed = time.time() - self._last_request_time
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
        self._last_request_time = time.time()
        self._request_count += 1

    async def _retry_with_backoff(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """Execute a function with exponential backoff on failure."""
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}), "
                        f"retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
        raise last_error


class ProviderRegistry:
    """
    Registry for provider instances.

    Supports:
    - Named provider registration
    - Default provider selection
    - Provider pooling for load balancing
    """

    _providers: Dict[str, BaseProvider] = {}
    _default: Optional[str] = None

    @classmethod
    def register(
        cls,
        name: str,
        provider: BaseProvider,
        set_default: bool = False,
    ) -> None:
        """Register a provider instance."""
        cls._providers[name] = provider
        if set_default or cls._default is None:
            cls._default = name

    @classmethod
    def get(cls, name: Optional[str] = None) -> BaseProvider:
        """Get a provider by name or the default."""
        name = name or cls._default
        if name is None:
            raise ValueError("No provider registered")
        if name not in cls._providers:
            raise ValueError(f"Provider '{name}' not found")
        return cls._providers[name]

    @classmethod
    def list_providers(cls) -> List[str]:
        """List all registered provider names."""
        return list(cls._providers.keys())

    @classmethod
    def remove(cls, name: str) -> None:
        """Remove a provider from the registry."""
        if name in cls._providers:
            del cls._providers[name]
            if cls._default == name:
                cls._default = next(iter(cls._providers), None)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered providers."""
        cls._providers.clear()
        cls._default = None
