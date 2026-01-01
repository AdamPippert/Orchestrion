"""
Vector store nodes for RAG and semantic search.
"""

from __future__ import annotations
import math
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from ryven.node_env import *


class VectorStoreNode(Node):
    """
    In-memory vector store for embeddings.

    Stores embeddings with associated metadata for
    semantic search and retrieval.

    Inputs:
        - action: 'add', 'search', 'delete', 'clear'
        - embedding: Embedding vector to add
        - text: Original text for the embedding
        - metadata: Additional metadata
        - query_embedding: Embedding to search for
        - top_k: Number of results to return
        - exec: Trigger

    Outputs:
        - results: Search results with scores
        - count: Number of items in store
        - done: Exec
    """

    title = 'Vector Store'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='action'),
        NodeInputType(label='embedding'),
        NodeInputType(label='text'),
        NodeInputType(label='metadata'),
        NodeInputType(label='query_embedding'),
        NodeInputType(label='top_k', default=5),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='results'),
        NodeOutputType(label='count'),
        NodeOutputType(type_='exec', label='done'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self._store: List[Dict[str, Any]] = []

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def update_event(self, inp=-1):
        if inp != 6:
            return

        action = 'search'
        action_input = self.input(0)
        if action_input and action_input.payload:
            action = str(action_input.payload).lower()

        if action == 'add':
            emb_input = self.input(1)
            text_input = self.input(2)
            meta_input = self.input(3)

            if emb_input and emb_input.payload:
                embedding = emb_input.payload
                text = text_input.payload if text_input else ""
                metadata = meta_input.payload if meta_input else {}

                # Generate ID from content hash
                content_hash = hashlib.md5(str(text).encode()).hexdigest()[:8]

                self._store.append({
                    'id': f"doc_{len(self._store)}_{content_hash}",
                    'embedding': embedding,
                    'text': text,
                    'metadata': metadata,
                })

        elif action == 'search':
            query_input = self.input(4)
            if not query_input or not query_input.payload:
                self.set_output_val(0, Data([]))
                self.set_output_val(1, Data(len(self._store)))
                self.exec_output(2)
                return

            query_emb = query_input.payload
            top_k = 5
            k_input = self.input(5)
            if k_input and k_input.payload:
                top_k = int(k_input.payload)

            # Calculate similarities
            scored = []
            for item in self._store:
                score = self._cosine_similarity(query_emb, item['embedding'])
                scored.append((score, item))

            # Sort and return top-k
            scored.sort(key=lambda x: x[0], reverse=True)
            results = [
                {
                    'id': item['id'],
                    'text': item['text'],
                    'metadata': item['metadata'],
                    'score': score,
                }
                for score, item in scored[:top_k]
            ]

            self.set_output_val(0, Data(results))

        elif action == 'delete':
            # Delete by text match or ID
            text_input = self.input(2)
            if text_input and text_input.payload:
                target = str(text_input.payload)
                self._store = [
                    item for item in self._store
                    if item['text'] != target and item['id'] != target
                ]

        elif action == 'clear':
            self._store = []

        self.set_output_val(1, Data(len(self._store)))
        self.exec_output(2)

    def get_state(self) -> dict:
        return {'store': self._store}

    def set_state(self, data: dict, version):
        self._store = data.get('store', [])


class DocumentChunkerNode(Node):
    """
    Split documents into chunks for embedding.

    Implements various chunking strategies for
    optimal retrieval.

    Inputs:
        - text: Text to chunk
        - chunk_size: Target chunk size in characters
        - overlap: Overlap between chunks
        - strategy: 'fixed', 'sentence', 'paragraph'
        - exec: Trigger

    Outputs:
        - chunks: List of text chunks
        - count: Number of chunks
        - done: Exec
    """

    title = 'Document Chunker'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='text'),
        NodeInputType(label='chunk_size', default=500),
        NodeInputType(label='overlap', default=50),
        NodeInputType(label='strategy', default='fixed'),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='chunks'),
        NodeOutputType(label='count'),
        NodeOutputType(type_='exec', label='done'),
    ]

    def _chunk_fixed(self, text: str, size: int, overlap: int) -> List[str]:
        """Fixed-size chunking with overlap."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start = end - overlap
        return chunks

    def _chunk_sentence(self, text: str, size: int, overlap: int) -> List[str]:
        """Sentence-based chunking."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            if current_size + len(sentence) > size and current_chunk:
                chunks.append(' '.join(current_chunk))
                # Keep last sentence for overlap
                current_chunk = current_chunk[-1:] if overlap > 0 else []
                current_size = sum(len(s) for s in current_chunk)

            current_chunk.append(sentence)
            current_size += len(sentence)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _chunk_paragraph(self, text: str, size: int, overlap: int) -> List[str]:
        """Paragraph-based chunking."""
        paragraphs = text.split('\n\n')

        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if current_size + len(para) > size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_size = 0

            current_chunk.append(para)
            current_size += len(para)

        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks

    def update_event(self, inp=-1):
        if inp != 4:
            return

        text_input = self.input(0)
        if not text_input or not text_input.payload:
            return

        text = str(text_input.payload)

        chunk_size = 500
        cs_input = self.input(1)
        if cs_input and cs_input.payload:
            chunk_size = int(cs_input.payload)

        overlap = 50
        ov_input = self.input(2)
        if ov_input and ov_input.payload:
            overlap = int(ov_input.payload)

        strategy = 'fixed'
        strat_input = self.input(3)
        if strat_input and strat_input.payload:
            strategy = str(strat_input.payload).lower()

        if strategy == 'sentence':
            chunks = self._chunk_sentence(text, chunk_size, overlap)
        elif strategy == 'paragraph':
            chunks = self._chunk_paragraph(text, chunk_size, overlap)
        else:
            chunks = self._chunk_fixed(text, chunk_size, overlap)

        self.set_output_val(0, Data(chunks))
        self.set_output_val(1, Data(len(chunks)))
        self.exec_output(2)


class RetrievalNode(Node):
    """
    Retrieve relevant context for a query.

    Combines vector search with optional reranking.

    Inputs:
        - query: Text query
        - vector_store: Vector store to search
        - embedder: Provider for query embedding
        - top_k: Number of results
        - min_score: Minimum similarity score
        - exec: Trigger

    Outputs:
        - context: Combined relevant text
        - documents: Individual retrieved documents
        - scores: Relevance scores
        - done: Exec
    """

    title = 'Retrieval'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='query'),
        NodeInputType(label='results'),
        NodeInputType(label='top_k', default=3),
        NodeInputType(label='min_score', default=0.5),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='context'),
        NodeOutputType(label='documents'),
        NodeOutputType(label='scores'),
        NodeOutputType(type_='exec', label='done'),
    ]

    def update_event(self, inp=-1):
        if inp != 4:
            return

        results_input = self.input(1)
        if not results_input or not results_input.payload:
            self.set_output_val(0, Data(""))
            self.set_output_val(1, Data([]))
            self.set_output_val(2, Data([]))
            self.exec_output(3)
            return

        results = results_input.payload
        if not isinstance(results, list):
            results = [results]

        top_k = 3
        k_input = self.input(2)
        if k_input and k_input.payload:
            top_k = int(k_input.payload)

        min_score = 0.5
        ms_input = self.input(3)
        if ms_input and ms_input.payload is not None:
            min_score = float(ms_input.payload)

        # Filter by score
        filtered = [r for r in results if r.get('score', 0) >= min_score][:top_k]

        # Extract texts
        documents = [r.get('text', '') for r in filtered]
        scores = [r.get('score', 0) for r in filtered]

        # Combine into context
        context = "\n\n---\n\n".join(documents)

        self.set_output_val(0, Data(context))
        self.set_output_val(1, Data(documents))
        self.set_output_val(2, Data(scores))
        self.exec_output(3)


class HybridSearchNode(Node):
    """
    Combine vector and keyword search.

    Uses both semantic similarity and keyword matching
    for improved retrieval.

    Inputs:
        - query: Search query
        - vector_results: Results from vector search
        - keyword_results: Results from keyword search
        - vector_weight: Weight for vector results (0-1)
        - top_k: Number of final results
        - exec: Trigger

    Outputs:
        - results: Combined ranked results
        - done: Exec
    """

    title = 'Hybrid Search'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='query'),
        NodeInputType(label='vector_results'),
        NodeInputType(label='keyword_results'),
        NodeInputType(label='vector_weight', default=0.7),
        NodeInputType(label='top_k', default=5),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='results'),
        NodeOutputType(type_='exec', label='done'),
    ]

    def update_event(self, inp=-1):
        if inp != 5:
            return

        vector_input = self.input(1)
        keyword_input = self.input(2)

        vector_results = vector_input.payload if vector_input and vector_input.payload else []
        keyword_results = keyword_input.payload if keyword_input and keyword_input.payload else []

        vector_weight = 0.7
        vw_input = self.input(3)
        if vw_input and vw_input.payload is not None:
            vector_weight = float(vw_input.payload)

        keyword_weight = 1 - vector_weight

        top_k = 5
        k_input = self.input(4)
        if k_input and k_input.payload:
            top_k = int(k_input.payload)

        # Combine scores
        combined: Dict[str, Dict[str, Any]] = {}

        for r in vector_results:
            doc_id = r.get('id', str(r.get('text', '')))
            combined[doc_id] = {
                **r,
                'vector_score': r.get('score', 0),
                'keyword_score': 0,
            }

        for r in keyword_results:
            doc_id = r.get('id', str(r.get('text', '')))
            if doc_id in combined:
                combined[doc_id]['keyword_score'] = r.get('score', 0)
            else:
                combined[doc_id] = {
                    **r,
                    'vector_score': 0,
                    'keyword_score': r.get('score', 0),
                }

        # Calculate final scores
        for doc_id, doc in combined.items():
            doc['score'] = (
                vector_weight * doc['vector_score'] +
                keyword_weight * doc['keyword_score']
            )

        # Sort and return
        results = sorted(combined.values(), key=lambda x: x['score'], reverse=True)[:top_k]

        self.set_output_val(0, Data(results))
        self.exec_output(1)


# Export vector store nodes
vector_store_nodes = [
    VectorStoreNode,
    DocumentChunkerNode,
    RetrievalNode,
    HybridSearchNode,
]
