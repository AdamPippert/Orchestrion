"""
RAG (Retrieval-Augmented Generation) Workflow Example

Demonstrates building a RAG pipeline with:
- Document loading and chunking
- Vector store indexing
- Semantic retrieval
- Context-augmented generation

This example shows the complete flow from documents to answers.
"""

import asyncio
from typing import List, Optional


class SimpleRAGPipeline:
    """
    A simple RAG pipeline demonstrating the Orchestrion node pattern.

    Node Flow:
        [Documents] --> [Chunker] --> [Embedder] --> [Vector Store]
                                                           |
        [Query] --> [Query Embedder] --> [Retriever] <-----+
                                              |
                            [Context Builder] <-+
                                     |
        [System Prompt] --> [Message Builder] --> [LLM] --> [Response]
    """

    def __init__(self, provider, embedding_provider=None):
        """Initialize the RAG pipeline."""
        self.provider = provider
        self.embedding_provider = embedding_provider or provider
        self.vector_store = []  # Simple in-memory store
        self.documents = []

    async def add_documents(
        self,
        texts: List[str],
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """
        Add documents to the vector store.

        Simulates:
        - DocumentChunkerNode
        - EmbeddingNode
        - VectorStoreNode (add operation)
        """
        print(f"Adding {len(texts)} documents...")

        for text in texts:
            # Chunk the document
            chunks = self._chunk_text(text, chunk_size, chunk_overlap)

            for chunk in chunks:
                # Generate embedding (simplified)
                embedding = await self._embed_text(chunk)

                # Store in vector store
                self.vector_store.append({
                    'text': chunk,
                    'embedding': embedding,
                })
                self.documents.append(chunk)

        print(f"Indexed {len(self.vector_store)} chunks")

    async def query(
        self,
        question: str,
        top_k: int = 3,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Query the RAG pipeline.

        Simulates:
        - EmbeddingNode (for query)
        - RetrievalNode
        - ContextWindowNode
        - ChatNode
        """
        print(f"\nQuery: {question}")

        # 1. Embed the query
        query_embedding = await self._embed_text(question)

        # 2. Retrieve relevant chunks
        relevant_chunks = self._retrieve(query_embedding, top_k)
        print(f"Retrieved {len(relevant_chunks)} relevant chunks")

        # 3. Build context
        context = "\n\n".join([
            f"[Document {i+1}]\n{chunk['text']}"
            for i, chunk in enumerate(relevant_chunks)
        ])

        # 4. Build prompt with context
        if system_prompt is None:
            system_prompt = """You are a helpful assistant that answers questions based on the provided context.
If the context doesn't contain relevant information, say so.
Always cite which document(s) you used to answer."""

        user_message = f"""Context:
{context}

Question: {question}

Answer based on the context above:"""

        # 5. Generate response
        from agent_nodes.providers import Message

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_message),
        ]

        response = await self.provider.chat(
            messages=messages,
            temperature=0.3,  # Lower temp for factual responses
        )

        print(f"\nAnswer: {response.content}")
        return response.content

    def _chunk_text(
        self,
        text: str,
        size: int,
        overlap: int,
    ) -> List[str]:
        """
        Chunk text into smaller pieces.

        Simulates DocumentChunkerNode with fixed-size strategy.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start += size - overlap
        return chunks

    async def _embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for text.

        Simulates EmbeddingNode.
        In production, use actual embedding API.
        """
        # Simple hash-based pseudo-embedding for demo
        import hashlib
        hash_bytes = hashlib.md5(text.encode()).digest()
        return [b / 255.0 for b in hash_bytes]

    def _retrieve(
        self,
        query_embedding: List[float],
        top_k: int,
    ) -> List[dict]:
        """
        Retrieve most similar chunks.

        Simulates RetrievalNode with cosine similarity.
        """
        def cosine_similarity(a, b):
            import math
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 0
            return dot / (norm_a * norm_b)

        # Score all chunks
        scored = [
            {
                **chunk,
                'score': cosine_similarity(query_embedding, chunk['embedding'])
            }
            for chunk in self.vector_store
        ]

        # Sort by score and return top_k
        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored[:top_k]


# Example node workflow visualization
def visualize_rag_workflow():
    """
    Visualize the RAG workflow as it would appear in Orchestrion.
    """
    workflow = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    RAG Workflow in Orchestrion                ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║                                                               ║
    ║  ┌─────────────┐                                              ║
    ║  │  Documents  │                                              ║
    ║  └──────┬──────┘                                              ║
    ║         │                                                     ║
    ║         ▼                                                     ║
    ║  ┌─────────────┐     ┌─────────────┐                          ║
    ║  │   Chunker   │────▶│  Embedder   │                          ║
    ║  │ (500 chars) │     │ (text→vec)  │                          ║
    ║  └─────────────┘     └──────┬──────┘                          ║
    ║                             │                                 ║
    ║                             ▼                                 ║
    ║                      ┌─────────────┐                          ║
    ║                      │Vector Store │◀─────────────┐           ║
    ║                      │  (add/get)  │              │           ║
    ║                      └──────┬──────┘              │           ║
    ║                             │                     │           ║
    ║  ┌─────────────┐            │                     │           ║
    ║  │    Query    │            │                     │           ║
    ║  └──────┬──────┘            │                     │           ║
    ║         │                   │                     │           ║
    ║         ▼                   │                     │           ║
    ║  ┌─────────────┐            │                     │           ║
    ║  │  Embedder   │────────────┘                     │           ║
    ║  │ (query→vec) │                                  │           ║
    ║  └──────┬──────┘                                  │           ║
    ║         │                                         │           ║
    ║         ▼                                         │           ║
    ║  ┌─────────────┐     ┌─────────────┐              │           ║
    ║  │  Retriever  │────▶│   Context   │              │           ║
    ║  │  (top-k=3)  │     │   Builder   │              │           ║
    ║  └─────────────┘     └──────┬──────┘              │           ║
    ║                             │                     │           ║
    ║  ┌─────────────┐            │                     │           ║
    ║  │   System    │            │                     │           ║
    ║  │   Prompt    │────┐       │                     │           ║
    ║  └─────────────┘    │       │                     │           ║
    ║                     ▼       ▼                     │           ║
    ║                ┌─────────────────┐                │           ║
    ║                │  Message List   │                │           ║
    ║                │ (system + user) │                │           ║
    ║                └────────┬────────┘                │           ║
    ║                         │                         │           ║
    ║                         ▼                         │           ║
    ║                  ┌─────────────┐                  │           ║
    ║                  │    Chat     │                  │           ║
    ║                  │   (LLM)     │                  │           ║
    ║                  └──────┬──────┘                  │           ║
    ║                         │                         │           ║
    ║                         ▼                         │           ║
    ║                  ┌─────────────┐                  │           ║
    ║                  │   Answer    │──────────────────┘           ║
    ║                  └─────────────┘     (feedback loop)          ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(workflow)


async def demo_rag_pipeline():
    """
    Demonstrate the RAG pipeline with sample documents.
    """
    from unittest.mock import AsyncMock, MagicMock

    # Create mock provider for demo
    mock_provider = MagicMock()
    mock_provider.chat = AsyncMock(return_value=MagicMock(
        content="Based on Document 1 and 2, Python is a programming language created by Guido van Rossum. It is known for its simple syntax and readability."
    ))

    # Create RAG pipeline
    rag = SimpleRAGPipeline(provider=mock_provider)

    # Sample documents
    documents = [
        """Python is a high-level, general-purpose programming language.
Its design philosophy emphasizes code readability with the use of
significant indentation. Python is dynamically typed and garbage-collected.
It supports multiple programming paradigms, including structured, object-oriented,
and functional programming.""",

        """Python was conceived in the late 1980s by Guido van Rossum at Centrum
Wiskunde & Informatica (CWI) in the Netherlands as a successor to the ABC
programming language. Its implementation began in December 1989. Van Rossum
shouldered sole responsibility for the project until 2018.""",

        """Python consistently ranks as one of the most popular programming languages.
Large organizations that use Python include Google, NASA, Facebook, and Amazon.
It is widely used in web development, data science, artificial intelligence,
scientific computing, and automation.""",
    ]

    # Add documents
    await rag.add_documents(documents, chunk_size=200, chunk_overlap=20)

    # Query
    await rag.query("What is Python and who created it?")


if __name__ == "__main__":
    print("=" * 60)
    print("RAG Workflow Example")
    print("=" * 60)

    print("\n1. Workflow Visualization:")
    visualize_rag_workflow()

    print("\n2. Pipeline Demo:")
    asyncio.run(demo_rag_pipeline())
