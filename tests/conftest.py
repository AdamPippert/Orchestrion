"""
Pytest configuration for Orchestrion tests.
"""

import pytest
import sys
import os
import types

# Add necessary paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../ryven-editor/ryven/example_nodes'))

# Mock ryven module before any agent_nodes imports
if 'ryven' not in sys.modules:
    mock_ryven = types.ModuleType('ryven')
    mock_node_env = types.ModuleType('ryven.node_env')

    class MockNode:
        title = ''
        version = ''
        init_inputs = []
        init_outputs = []
        def __init__(self, params=None): pass
        def input(self, i): return None
        def set_output_val(self, i, v): pass

    mock_node_env.Node = MockNode
    mock_node_env.NodeInputType = lambda **kwargs: None
    mock_node_env.NodeOutputType = lambda **kwargs: None
    mock_node_env.Data = lambda v: type('Data', (), {'payload': v})()
    mock_node_env.on_gui_load = lambda f: f
    mock_node_env.export_nodes = lambda n: None

    sys.modules['ryven'] = mock_ryven
    sys.modules['ryven.node_env'] = mock_node_env


@pytest.fixture
def mock_provider():
    """Create a mock LLM provider for testing."""
    from unittest.mock import MagicMock, AsyncMock
    from agent_nodes.providers.base import BaseProvider, CompletionResponse

    provider = MagicMock(spec=BaseProvider)
    provider.initialize = AsyncMock()
    provider.close = AsyncMock()
    provider.chat = AsyncMock(return_value=CompletionResponse(
        content="Test response",
        model="test-model",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    ))
    return provider


@pytest.fixture
def sample_messages():
    """Create sample messages for testing."""
    from agent_nodes.providers.base import Message

    return [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Hello!"),
    ]


@pytest.fixture
def sample_embedding():
    """Create a sample embedding vector."""
    import random
    random.seed(42)
    return [random.random() for _ in range(384)]


@pytest.fixture
def sample_documents():
    """Create sample documents for RAG testing."""
    return [
        "Python is a programming language known for its simple syntax.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand text.",
        "Deep learning uses neural networks with multiple layers.",
        "Data science combines statistics with programming skills.",
    ]
