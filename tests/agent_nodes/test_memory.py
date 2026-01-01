"""
Tests for memory and RAG nodes.
"""

import pytest
import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../ryven-editor/ryven/example_nodes'))


class TestVectorStore:
    """Tests for vector store operations."""

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        from agent_nodes.memory.vector_store import VectorStoreNode

        node = VectorStoreNode.__new__(VectorStoreNode)

        # Identical vectors
        sim = node._cosine_similarity([1, 0, 0], [1, 0, 0])
        assert sim == 1.0

        # Orthogonal vectors
        sim = node._cosine_similarity([1, 0, 0], [0, 1, 0])
        assert sim == 0.0

        # Opposite vectors
        sim = node._cosine_similarity([1, 0, 0], [-1, 0, 0])
        assert sim == -1.0

        # Similar vectors
        sim = node._cosine_similarity([1, 1, 0], [1, 0, 0])
        expected = 1 / math.sqrt(2)
        assert abs(sim - expected) < 0.001


class TestDocumentChunker:
    """Tests for document chunking."""

    def test_fixed_chunking(self):
        """Test fixed-size chunking."""
        from agent_nodes.memory.vector_store import DocumentChunkerNode

        node = DocumentChunkerNode.__new__(DocumentChunkerNode)

        text = "A" * 1000
        chunks = node._chunk_fixed(text, size=200, overlap=50)

        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk) <= 200

    def test_sentence_chunking(self):
        """Test sentence-based chunking."""
        from agent_nodes.memory.vector_store import DocumentChunkerNode

        node = DocumentChunkerNode.__new__(DocumentChunkerNode)

        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = node._chunk_sentence(text, size=50, overlap=0)

        assert len(chunks) > 0

    def test_paragraph_chunking(self):
        """Test paragraph-based chunking."""
        from agent_nodes.memory.vector_store import DocumentChunkerNode

        node = DocumentChunkerNode.__new__(DocumentChunkerNode)

        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = node._chunk_paragraph(text, size=100, overlap=0)

        assert len(chunks) > 0


class TestContextWindow:
    """Tests for context window management."""

    def test_token_estimation(self):
        """Test token estimation."""
        from agent_nodes.memory.context import ContextWindowNode

        node = ContextWindowNode.__new__(ContextWindowNode)

        # Roughly 4 chars per token
        tokens = node._estimate_tokens("Hello world")
        assert tokens == 2  # 11 chars / 4 ≈ 2

        tokens = node._estimate_tokens("A" * 100)
        assert tokens == 25  # 100 / 4 = 25


class TestContextCompressor:
    """Tests for context compression."""

    def test_extractive_compression(self):
        """Test extractive compression."""
        from agent_nodes.memory.context import ContextCompressorNode

        node = ContextCompressorNode.__new__(ContextCompressorNode)

        text = "Important first sentence. " * 10
        compressed = node._extractive_compress(text, ratio=0.3)

        assert len(compressed) < len(text)

    def test_keyword_compression(self):
        """Test keyword-based compression."""
        from agent_nodes.memory.context import ContextCompressorNode

        node = ContextCompressorNode.__new__(ContextCompressorNode)

        text = 'The "important quote" from Company Name about 50% growth.'
        compressed = node._keyword_compress(text, ratio=0.5)

        assert len(compressed) > 0

    def test_first_last_compression(self):
        """Test first/last compression."""
        from agent_nodes.memory.context import ContextCompressorNode

        node = ContextCompressorNode.__new__(ContextCompressorNode)

        text = "A" * 100 + "B" * 100 + "C" * 100
        compressed = node._first_last_compress(text, ratio=0.5)

        assert len(compressed) < len(text)
        assert "A" in compressed
        assert "C" in compressed


class TestRelevanceFilter:
    """Tests for relevance filtering."""

    def test_simple_relevance(self):
        """Test simple relevance scoring."""
        from agent_nodes.memory.context import RelevanceFilterNode

        node = RelevanceFilterNode.__new__(RelevanceFilterNode)

        # High relevance
        score = node._simple_relevance("machine learning python", "python")
        assert score > 0

        # Low relevance
        score = node._simple_relevance("something completely different", "python")
        assert score == 0


class TestKnowledgeBase:
    """Tests for knowledge base."""

    def test_knowledge_base_node_exists(self):
        """Verify KnowledgeBaseNode is properly defined."""
        from agent_nodes.memory.knowledge import KnowledgeBaseNode

        assert hasattr(KnowledgeBaseNode, 'title')
        assert KnowledgeBaseNode.title == 'Knowledge Base'


class TestFactStore:
    """Tests for fact storage."""

    def test_fact_store_node_exists(self):
        """Verify FactStoreNode is properly defined."""
        from agent_nodes.memory.knowledge import FactStoreNode

        assert hasattr(FactStoreNode, 'title')
        assert FactStoreNode.title == 'Fact Store'


class TestEntityMemory:
    """Tests for entity memory."""

    def test_entity_memory_node_exists(self):
        """Verify EntityMemoryNode is properly defined."""
        from agent_nodes.memory.knowledge import EntityMemoryNode

        assert hasattr(EntityMemoryNode, 'title')
        assert EntityMemoryNode.title == 'Entity Memory'
