"""
Embedding nodes for vector operations and semantic search.
"""

from __future__ import annotations
import asyncio
import math
from typing import List, Optional
from ryven.node_env import *

from ..providers.base import BaseProvider, ProviderRegistry


class EmbeddingNode(Node):
    """
    Generate embeddings for text using an LLM provider.

    Embeddings are dense vector representations of text that capture
    semantic meaning, useful for similarity search and RAG.

    Inputs:
        - provider: LLM provider with embedding support
        - text: Text to embed (string or list of strings)
        - model: Embedding model to use (optional)

    Outputs:
        - embedding: Vector representation(s) of the text
        - dimensions: Number of dimensions in the embedding
    """

    title = 'Embed Text'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='provider'),
        NodeInputType(label='text'),
        NodeInputType(label='model'),
    ]
    init_outputs = [
        NodeOutputType(label='embedding'),
        NodeOutputType(label='dimensions'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self._loop = None

    def _ensure_loop(self):
        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop

    def get_provider(self) -> Optional[BaseProvider]:
        provider_input = self.input(0)
        if provider_input and provider_input.payload:
            return provider_input.payload
        try:
            return ProviderRegistry.get()
        except ValueError:
            return None

    def update_event(self, inp=-1):
        provider = self.get_provider()
        if not provider:
            self.set_output_val(0, Data("Error: No provider configured"))
            return

        text_input = self.input(1)
        if not text_input or not text_input.payload:
            return

        texts = text_input.payload
        if isinstance(texts, str):
            texts = [texts]

        model = None
        model_input = self.input(2)
        if model_input and model_input.payload:
            model = model_input.payload

        loop = self._ensure_loop()

        async def do_embed():
            await provider.initialize()
            try:
                return await provider.embed(texts, model=model)
            finally:
                await provider.close()

        try:
            embeddings = loop.run_until_complete(do_embed())
            # Return single embedding or list
            if len(embeddings) == 1:
                self.set_output_val(0, Data(embeddings[0]))
                self.set_output_val(1, Data(len(embeddings[0])))
            else:
                self.set_output_val(0, Data(embeddings))
                self.set_output_val(1, Data(len(embeddings[0]) if embeddings else 0))
        except Exception as e:
            self.set_output_val(0, Data(f"Error: {str(e)}"))


class SimilarityNode(Node):
    """
    Calculate cosine similarity between two embeddings.

    Useful for comparing semantic similarity between texts.
    Returns a value between -1 and 1, where 1 means identical.

    Inputs:
        - embedding_a: First embedding vector
        - embedding_b: Second embedding vector

    Outputs:
        - similarity: Cosine similarity score (-1 to 1)
        - distance: Cosine distance (1 - similarity)
    """

    title = 'Similarity'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='embedding_a'),
        NodeInputType(label='embedding_b'),
    ]
    init_outputs = [
        NodeOutputType(label='similarity'),
        NodeOutputType(label='distance'),
    ]

    def update_event(self, inp=-1):
        a_input = self.input(0)
        b_input = self.input(1)

        if not a_input or not a_input.payload:
            return
        if not b_input or not b_input.payload:
            return

        a = a_input.payload
        b = b_input.payload

        if not isinstance(a, list) or not isinstance(b, list):
            self.set_output_val(0, Data("Error: Inputs must be embedding vectors"))
            return

        if len(a) != len(b):
            self.set_output_val(0, Data("Error: Embeddings must have same dimensions"))
            return

        # Compute cosine similarity
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            similarity = 0.0
        else:
            similarity = dot_product / (norm_a * norm_b)

        self.set_output_val(0, Data(similarity))
        self.set_output_val(1, Data(1 - similarity))


class TopKSimilarNode(Node):
    """
    Find the top-K most similar items from a collection of embeddings.

    Inputs:
        - query_embedding: The embedding to search for
        - embeddings: List of embeddings to search through
        - items: Corresponding items/labels for each embedding
        - k: Number of results to return

    Outputs:
        - results: Top-K most similar items with scores
        - indices: Indices of the top-K items
        - scores: Similarity scores
    """

    title = 'Top-K Similar'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='query_embedding'),
        NodeInputType(label='embeddings'),
        NodeInputType(label='items'),
        NodeInputType(label='k', default=5),
    ]
    init_outputs = [
        NodeOutputType(label='results'),
        NodeOutputType(label='indices'),
        NodeOutputType(label='scores'),
    ]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def update_event(self, inp=-1):
        query_input = self.input(0)
        embeddings_input = self.input(1)
        items_input = self.input(2)
        k_input = self.input(3)

        if not query_input or not query_input.payload:
            return
        if not embeddings_input or not embeddings_input.payload:
            return

        query = query_input.payload
        embeddings = embeddings_input.payload

        items = None
        if items_input and items_input.payload:
            items = items_input.payload

        k = 5
        if k_input and k_input.payload:
            k = int(k_input.payload)

        # Calculate similarities
        similarities = []
        for i, emb in enumerate(embeddings):
            sim = self._cosine_similarity(query, emb)
            similarities.append((i, sim))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Get top-K
        top_k = similarities[:k]
        indices = [x[0] for x in top_k]
        scores = [x[1] for x in top_k]

        results = []
        for idx, score in top_k:
            item = items[idx] if items and idx < len(items) else idx
            results.append({'item': item, 'index': idx, 'score': score})

        self.set_output_val(0, Data(results))
        self.set_output_val(1, Data(indices))
        self.set_output_val(2, Data(scores))


# Export all embedding nodes
embedding_nodes = [
    EmbeddingNode,
    SimilarityNode,
    TopKSimilarNode,
]
