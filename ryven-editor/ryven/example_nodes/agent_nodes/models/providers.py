"""
Provider configuration nodes for connecting to LLM services.
"""

from __future__ import annotations
from ryven.node_env import *

from ..providers.base import ProviderConfig, ProviderType, ProviderRegistry
from ..providers.openai_compat import OpenAICompatibleProvider
from ..providers.anthropic import AnthropicProvider


class ProviderConfigNode(Node):
    """
    Configure a generic LLM provider.

    Creates a provider configuration that can be passed to
    chat and embedding nodes.

    Inputs:
        - provider_type: Type of provider (openai, anthropic, vllm, etc.)
        - api_key: API key for authentication
        - base_url: Base URL for the API (for custom endpoints)
        - model: Default model to use
        - timeout: Request timeout in seconds
        - name: Name to register this provider under

    Outputs:
        - provider: Configured provider instance
        - config: Raw configuration object
    """

    title = 'Provider Config'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='provider_type'),
        NodeInputType(label='api_key'),
        NodeInputType(label='base_url'),
        NodeInputType(label='model'),
        NodeInputType(label='timeout', default=60.0),
        NodeInputType(label='name'),
    ]
    init_outputs = [
        NodeOutputType(label='provider'),
        NodeOutputType(label='config'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self._provider = None

    def update_event(self, inp=-1):
        # Get provider type
        type_input = self.input(0)
        if not type_input or not type_input.payload:
            return

        provider_type_str = str(type_input.payload).lower()

        # Map string to ProviderType enum
        type_map = {
            'openai': ProviderType.OPENAI,
            'anthropic': ProviderType.ANTHROPIC,
            'vllm': ProviderType.VLLM,
            'ramalama': ProviderType.RAMALAMA,
            'ollama': ProviderType.OLLAMA,
            'llamacpp': ProviderType.LLAMACPP,
            'custom': ProviderType.CUSTOM,
        }

        provider_type = type_map.get(provider_type_str)
        if not provider_type:
            self.set_output_val(0, Data(f"Error: Unknown provider type '{provider_type_str}'"))
            return

        # Get optional parameters
        api_key = None
        key_input = self.input(1)
        if key_input and key_input.payload:
            api_key = str(key_input.payload)

        base_url = None
        url_input = self.input(2)
        if url_input and url_input.payload:
            base_url = str(url_input.payload)

        model = ""
        model_input = self.input(3)
        if model_input and model_input.payload:
            model = str(model_input.payload)

        timeout = 60.0
        timeout_input = self.input(4)
        if timeout_input and timeout_input.payload:
            timeout = float(timeout_input.payload)

        name = None
        name_input = self.input(5)
        if name_input and name_input.payload:
            name = str(name_input.payload)

        # Create config
        config = ProviderConfig(
            provider_type=provider_type,
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
        )

        # Create appropriate provider
        if provider_type == ProviderType.ANTHROPIC:
            provider = AnthropicProvider(config)
        else:
            provider = OpenAICompatibleProvider(config)

        # Register if name provided
        if name:
            ProviderRegistry.register(name, provider)

        self._provider = provider
        self.set_output_val(0, Data(provider))
        self.set_output_val(1, Data(config))


class OpenAIProviderNode(Node):
    """
    Quick setup for OpenAI API.

    Simplified configuration specifically for OpenAI's API.

    Inputs:
        - api_key: OpenAI API key (or uses OPENAI_API_KEY env var)
        - model: Model to use (default: gpt-4o)
        - register_as: Name to register this provider under

    Outputs:
        - provider: Configured OpenAI provider
    """

    title = 'OpenAI Provider'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='api_key'),
        NodeInputType(label='model', default='gpt-4o'),
        NodeInputType(label='register_as'),
    ]
    init_outputs = [
        NodeOutputType(label='provider'),
    ]

    def update_event(self, inp=-1):
        api_key = None
        key_input = self.input(0)
        if key_input and key_input.payload:
            api_key = str(key_input.payload)

        model = 'gpt-4o'
        model_input = self.input(1)
        if model_input and model_input.payload:
            model = str(model_input.payload)

        register_as = None
        reg_input = self.input(2)
        if reg_input and reg_input.payload:
            register_as = str(reg_input.payload)

        config = ProviderConfig(
            provider_type=ProviderType.OPENAI,
            api_key=api_key,
            model=model,
        )
        provider = OpenAICompatibleProvider(config)

        if register_as:
            ProviderRegistry.register(register_as, provider)

        self.set_output_val(0, Data(provider))


class AnthropicProviderNode(Node):
    """
    Quick setup for Anthropic Claude API.

    Simplified configuration for Claude models.

    Inputs:
        - api_key: Anthropic API key (or uses ANTHROPIC_API_KEY env var)
        - model: Model to use (default: claude-sonnet-4-20250514)
        - register_as: Name to register this provider under

    Outputs:
        - provider: Configured Anthropic provider
    """

    title = 'Anthropic Provider'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='api_key'),
        NodeInputType(label='model', default='claude-sonnet-4-20250514'),
        NodeInputType(label='register_as'),
    ]
    init_outputs = [
        NodeOutputType(label='provider'),
    ]

    def update_event(self, inp=-1):
        api_key = None
        key_input = self.input(0)
        if key_input and key_input.payload:
            api_key = str(key_input.payload)

        model = 'claude-sonnet-4-20250514'
        model_input = self.input(1)
        if model_input and model_input.payload:
            model = str(model_input.payload)

        register_as = None
        reg_input = self.input(2)
        if reg_input and reg_input.payload:
            register_as = str(reg_input.payload)

        config = ProviderConfig(
            provider_type=ProviderType.ANTHROPIC,
            api_key=api_key,
            model=model,
        )
        provider = AnthropicProvider(config)

        if register_as:
            ProviderRegistry.register(register_as, provider)

        self.set_output_val(0, Data(provider))


class LocalProviderNode(Node):
    """
    Configure a local LLM provider (vLLM, RamaLama, Ollama, llama.cpp).

    Provides easy setup for locally-running inference servers
    that expose OpenAI-compatible APIs.

    Inputs:
        - runtime: Type of local runtime (vllm, ramalama, ollama, llamacpp)
        - base_url: URL of the local server
        - model: Model name/path
        - register_as: Name to register this provider under

    Outputs:
        - provider: Configured local provider
    """

    title = 'Local Provider'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='runtime'),
        NodeInputType(label='base_url'),
        NodeInputType(label='model'),
        NodeInputType(label='register_as'),
    ]
    init_outputs = [
        NodeOutputType(label='provider'),
    ]

    def update_event(self, inp=-1):
        runtime_input = self.input(0)
        if not runtime_input or not runtime_input.payload:
            return

        runtime = str(runtime_input.payload).lower()

        # Default URLs for each runtime
        default_urls = {
            'vllm': 'http://localhost:8000/v1',
            'ramalama': 'http://localhost:8080/v1',
            'ollama': 'http://localhost:11434/v1',
            'llamacpp': 'http://localhost:8080/v1',
        }

        type_map = {
            'vllm': ProviderType.VLLM,
            'ramalama': ProviderType.RAMALAMA,
            'ollama': ProviderType.OLLAMA,
            'llamacpp': ProviderType.LLAMACPP,
        }

        provider_type = type_map.get(runtime)
        if not provider_type:
            self.set_output_val(0, Data(f"Error: Unknown runtime '{runtime}'"))
            return

        base_url = default_urls.get(runtime)
        url_input = self.input(1)
        if url_input and url_input.payload:
            base_url = str(url_input.payload)

        model = ''
        model_input = self.input(2)
        if model_input and model_input.payload:
            model = str(model_input.payload)

        register_as = None
        reg_input = self.input(3)
        if reg_input and reg_input.payload:
            register_as = str(reg_input.payload)

        config = ProviderConfig(
            provider_type=provider_type,
            base_url=base_url,
            model=model,
        )
        provider = OpenAICompatibleProvider(config)

        if register_as:
            ProviderRegistry.register(register_as, provider)

        self.set_output_val(0, Data(provider))


# Export all provider nodes
provider_nodes = [
    ProviderConfigNode,
    OpenAIProviderNode,
    AnthropicProviderNode,
    LocalProviderNode,
]
