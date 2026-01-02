"""
Anthropic Claude provider.

Supports Claude 3.5, Claude 3 Opus, Sonnet, and Haiku models
with full feature support including:
- Chat completions
- Vision (image analysis)
- Tool/function calling
- Streaming
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


class AnthropicProvider(BaseProvider):
    """
    Provider for Anthropic's Claude models.

    Uses the native Anthropic SDK for optimal compatibility
    with Claude-specific features.
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._async_client: Any = None

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.ANTHROPIC

    async def initialize(self) -> None:
        """Initialize the Anthropic client."""
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            )

        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY or pass api_key."
            )

        self._async_client = AsyncAnthropic(
            api_key=api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=0,
        )

    async def close(self) -> None:
        """Close the client connection."""
        if self._async_client:
            await self._async_client.close()
            self._async_client = None

    def _convert_messages(
        self,
        messages: List[Message],
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Convert messages to Anthropic format.

        Anthropic uses a separate system parameter, so we extract it.
        Returns (system_prompt, messages).
        """
        system_prompt = None
        anthropic_messages = []

        for msg in messages:
            if msg.role == "system":
                # Combine multiple system messages if present
                if system_prompt:
                    system_prompt += "\n\n" + msg.content
                else:
                    system_prompt = msg.content if isinstance(msg.content, str) else str(msg.content)
            else:
                # Convert role names (OpenAI uses "assistant", Anthropic uses "assistant")
                role = msg.role
                if role == "tool":
                    role = "user"  # Tool results go in user messages

                content = msg.content

                # Handle tool results
                if msg.tool_call_id:
                    content = [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.content if isinstance(msg.content, str) else str(msg.content),
                        }
                    ]

                anthropic_messages.append({
                    "role": role,
                    "content": content,
                })

        return system_prompt, anthropic_messages

    def _convert_tools(
        self,
        tools: Optional[List[Dict[str, Any]]],
    ) -> Optional[List[Dict[str, Any]]]:
        """Convert OpenAI-style tools to Anthropic format."""
        if not tools:
            return None

        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })
        return anthropic_tools

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
        """Send a chat completion request to Claude."""
        await self._rate_limit()

        model = model or self.config.model or "claude-sonnet-4-20250514"
        max_tokens = max_tokens or 4096  # Anthropic requires max_tokens
        start_time = time.perf_counter()

        system_prompt, anthropic_messages = self._convert_messages(messages)

        request: Dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system_prompt:
            request["system"] = system_prompt
        if stop:
            request["stop_sequences"] = stop

        # Handle tools
        anthropic_tools = self._convert_tools(tools)
        if anthropic_tools:
            request["tools"] = anthropic_tools

        # Handle tool choice
        if tool_choice:
            if isinstance(tool_choice, str):
                if tool_choice == "auto":
                    request["tool_choice"] = {"type": "auto"}
                elif tool_choice == "required":
                    request["tool_choice"] = {"type": "any"}
                elif tool_choice == "none":
                    pass  # Don't send tools
            elif isinstance(tool_choice, dict):
                if "function" in tool_choice:
                    request["tool_choice"] = {
                        "type": "tool",
                        "name": tool_choice["function"]["name"],
                    }

        async def _do_request():
            return await self._async_client.messages.create(**request)

        response = await self._retry_with_backoff(_do_request)
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Extract content and tool calls
        content_parts = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": str(block.input) if not isinstance(block.input, str) else block.input,
                    },
                })

        return CompletionResponse(
            content="\n".join(content_parts),
            model=response.model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            tool_calls=tool_calls if tool_calls else None,
            latency_ms=latency_ms,
            finish_reason=response.stop_reason or "end_turn",
            provider_type=ProviderType.ANTHROPIC,
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
        """Stream a chat completion response from Claude."""
        await self._rate_limit()

        model = model or self.config.model or "claude-sonnet-4-20250514"
        max_tokens = max_tokens or 4096

        system_prompt, anthropic_messages = self._convert_messages(messages)

        request: Dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system_prompt:
            request["system"] = system_prompt

        anthropic_tools = self._convert_tools(tools)
        if anthropic_tools:
            request["tools"] = anthropic_tools

        async with self._async_client.messages.stream(**request) as stream:
            async for event in stream:
                if hasattr(event, 'type'):
                    if event.type == "content_block_delta":
                        if hasattr(event.delta, 'text'):
                            yield StreamChunk(content=event.delta.text)
                    elif event.type == "message_stop":
                        yield StreamChunk(finish_reason="end_turn")


# Convenience factory function
def create_anthropic_provider(
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-20250514",
    **kwargs,
) -> AnthropicProvider:
    """Create an Anthropic Claude provider."""
    config = ProviderConfig(
        provider_type=ProviderType.ANTHROPIC,
        api_key=api_key,
        model=model,
        **kwargs,
    )
    return AnthropicProvider(config)
