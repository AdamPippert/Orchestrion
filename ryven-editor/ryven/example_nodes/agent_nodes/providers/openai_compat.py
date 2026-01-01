"""
OpenAI-compatible provider.

Works with:
- OpenAI API
- vLLM (OpenAI-compatible server)
- RamaLama (OpenAI-compatible server)
- Ollama (OpenAI-compatible mode)
- LM Studio
- llama.cpp server
- Any OpenAI-compatible endpoint
"""

from __future__ import annotations
import os
import time
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from .base import (
    BaseProvider,
    ProviderConfig,
    ProviderType,
    Message,
    CompletionResponse,
    StreamChunk,
)

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(BaseProvider):
    """
    Provider for OpenAI and OpenAI-compatible APIs.

    This is the most versatile provider as many local inference
    servers implement the OpenAI API specification.
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._async_client: Any = None

    @property
    def provider_type(self) -> ProviderType:
        return self.config.provider_type

    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install openai>=1.0.0"
            )

        # Get API key from config or environment
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")

        # For local servers, API key might not be required
        if not api_key and self.config.provider_type in (
            ProviderType.VLLM,
            ProviderType.RAMALAMA,
            ProviderType.OLLAMA,
            ProviderType.LLAMACPP,
        ):
            api_key = "not-required"

        self._async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=0,  # We handle retries ourselves
        )

    async def close(self) -> None:
        """Close the client connection."""
        if self._async_client:
            await self._async_client.close()
            self._async_client = None

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
        """Send a chat completion request."""
        await self._rate_limit()

        model = model or self.config.model
        start_time = time.perf_counter()

        # Build request
        request = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
        }

        if max_tokens:
            request["max_tokens"] = max_tokens
        if tools:
            request["tools"] = tools
        if tool_choice:
            request["tool_choice"] = tool_choice
        if response_format:
            request["response_format"] = response_format
        if stop:
            request["stop"] = stop

        # Merge any extra kwargs
        request.update(kwargs)

        async def _do_request():
            return await self._async_client.chat.completions.create(**request)

        response = await self._retry_with_backoff(_do_request)
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Extract tool calls if present
        tool_calls = None
        if response.choices[0].message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in response.choices[0].message.tool_calls
            ]

        return CompletionResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
            tool_calls=tool_calls,
            latency_ms=latency_ms,
            finish_reason=response.choices[0].finish_reason or "stop",
            provider_type=self.provider_type,
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
        )

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
        """Stream a chat completion response."""
        await self._rate_limit()

        model = model or self.config.model

        request = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "stream": True,
        }

        if max_tokens:
            request["max_tokens"] = max_tokens
        if tools:
            request["tools"] = tools

        request.update(kwargs)

        stream = await self._async_client.chat.completions.create(**request)

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta

                # Handle tool calls in stream
                tool_calls = None
                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    tool_calls = [
                        {
                            "index": tc.index,
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": getattr(tc.function, 'name', ''),
                                "arguments": getattr(tc.function, 'arguments', ''),
                            },
                        }
                        for tc in delta.tool_calls
                    ]

                yield StreamChunk(
                    content=delta.content or "",
                    finish_reason=chunk.choices[0].finish_reason,
                    tool_calls=tool_calls,
                )

    async def embed(
        self,
        texts: List[str],
        *,
        model: Optional[str] = None,
        **kwargs,
    ) -> List[List[float]]:
        """Generate embeddings for texts."""
        await self._rate_limit()

        # Default embedding model
        model = model or "text-embedding-3-small"

        response = await self._async_client.embeddings.create(
            model=model,
            input=texts,
            **kwargs,
        )

        return [item.embedding for item in response.data]


# Convenience factory functions
def create_openai_provider(
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    **kwargs,
) -> OpenAICompatibleProvider:
    """Create an OpenAI provider."""
    config = ProviderConfig(
        provider_type=ProviderType.OPENAI,
        api_key=api_key,
        model=model,
        **kwargs,
    )
    return OpenAICompatibleProvider(config)


def create_vllm_provider(
    base_url: str = "http://localhost:8000/v1",
    model: str = "",
    **kwargs,
) -> OpenAICompatibleProvider:
    """Create a vLLM provider."""
    config = ProviderConfig(
        provider_type=ProviderType.VLLM,
        base_url=base_url,
        model=model,
        **kwargs,
    )
    return OpenAICompatibleProvider(config)


def create_ramalama_provider(
    base_url: str = "http://localhost:8080/v1",
    model: str = "",
    **kwargs,
) -> OpenAICompatibleProvider:
    """Create a RamaLama provider."""
    config = ProviderConfig(
        provider_type=ProviderType.RAMALAMA,
        base_url=base_url,
        model=model,
        **kwargs,
    )
    return OpenAICompatibleProvider(config)


def create_ollama_provider(
    base_url: str = "http://localhost:11434/v1",
    model: str = "llama3.2",
    **kwargs,
) -> OpenAICompatibleProvider:
    """Create an Ollama provider (OpenAI-compatible mode)."""
    config = ProviderConfig(
        provider_type=ProviderType.OLLAMA,
        base_url=base_url,
        model=model,
        **kwargs,
    )
    return OpenAICompatibleProvider(config)


def create_llamacpp_provider(
    base_url: str = "http://localhost:8080/v1",
    model: str = "",
    **kwargs,
) -> OpenAICompatibleProvider:
    """Create a llama.cpp server provider."""
    config = ProviderConfig(
        provider_type=ProviderType.LLAMACPP,
        base_url=base_url,
        model=model,
        **kwargs,
    )
    return OpenAICompatibleProvider(config)
