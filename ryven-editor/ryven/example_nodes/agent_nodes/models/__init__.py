"""
Model nodes for LLM inference.

Provides nodes for:
- Chat completions (with streaming support)
- Embeddings generation
- Vision/multimodal inputs
- Function/tool calling
"""

from .chat import (
    ChatNode,
    ChatStreamNode,
    SystemPromptNode,
    UserMessageNode,
    AssistantMessageNode,
    MessageListNode,
)
from .embeddings import EmbeddingNode, SimilarityNode
from .providers import (
    ProviderConfigNode,
    OpenAIProviderNode,
    AnthropicProviderNode,
    LocalProviderNode,
)

__all__ = [
    'ChatNode',
    'ChatStreamNode',
    'SystemPromptNode',
    'UserMessageNode',
    'AssistantMessageNode',
    'MessageListNode',
    'EmbeddingNode',
    'SimilarityNode',
    'ProviderConfigNode',
    'OpenAIProviderNode',
    'AnthropicProviderNode',
    'LocalProviderNode',
]
