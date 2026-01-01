"""
Context window management nodes.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from ryven.node_env import *


class ContextWindowNode(Node):
    """
    Manage context window for LLM conversations.

    Ensures messages fit within model context limits
    with smart truncation and summarization.

    Inputs:
        - messages: List of messages
        - max_tokens: Maximum tokens allowed
        - reserve_tokens: Tokens to reserve for response
        - strategy: 'truncate_old', 'summarize', 'smart'
        - exec: Trigger

    Outputs:
        - messages: Fitted messages
        - tokens_used: Estimated tokens in result
        - truncated: Whether truncation occurred
        - done: Exec
    """

    title = 'Context Window'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='messages'),
        NodeInputType(label='max_tokens', default=4096),
        NodeInputType(label='reserve_tokens', default=1024),
        NodeInputType(label='strategy', default='truncate_old'),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='messages'),
        NodeOutputType(label='tokens_used'),
        NodeOutputType(label='truncated'),
        NodeOutputType(type_='exec', label='done'),
    ]

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (4 chars per token)."""
        return len(text) // 4

    def _message_tokens(self, msg: Any) -> int:
        """Estimate tokens for a message."""
        if hasattr(msg, 'content'):
            content = msg.content
        elif isinstance(msg, dict):
            content = msg.get('content', '')
        else:
            content = str(msg)

        if isinstance(content, str):
            return self._estimate_tokens(content)
        return 100  # Default for non-text

    def update_event(self, inp=-1):
        if inp != 4:
            return

        messages_input = self.input(0)
        if not messages_input or not messages_input.payload:
            return

        messages = messages_input.payload
        if not isinstance(messages, list):
            messages = [messages]

        max_tokens = 4096
        mt_input = self.input(1)
        if mt_input and mt_input.payload:
            max_tokens = int(mt_input.payload)

        reserve = 1024
        res_input = self.input(2)
        if res_input and res_input.payload:
            reserve = int(res_input.payload)

        strategy = 'truncate_old'
        strat_input = self.input(3)
        if strat_input and strat_input.payload:
            strategy = str(strat_input.payload).lower()

        available = max_tokens - reserve
        truncated = False

        # Calculate current token usage
        current_tokens = sum(self._message_tokens(m) for m in messages)

        if current_tokens <= available:
            # Fits, no changes needed
            self.set_output_val(0, Data(messages))
            self.set_output_val(1, Data(current_tokens))
            self.set_output_val(2, Data(False))
            self.exec_output(3)
            return

        # Need to fit within limit
        truncated = True

        if strategy == 'truncate_old':
            # Keep system message and most recent messages
            result = []

            # Always keep system message if present
            if messages and (
                (hasattr(messages[0], 'role') and messages[0].role == 'system') or
                (isinstance(messages[0], dict) and messages[0].get('role') == 'system')
            ):
                result.append(messages[0])
                messages = messages[1:]

            # Add messages from the end until we hit limit
            remaining = available - sum(self._message_tokens(m) for m in result)
            selected = []

            for msg in reversed(messages):
                tokens = self._message_tokens(msg)
                if remaining >= tokens:
                    selected.insert(0, msg)
                    remaining -= tokens

            result.extend(selected)
            messages = result

        elif strategy == 'smart':
            # Keep system, first user message, and recent context
            result = []

            # System message
            if messages and (
                (hasattr(messages[0], 'role') and messages[0].role == 'system') or
                (isinstance(messages[0], dict) and messages[0].get('role') == 'system')
            ):
                result.append(messages[0])
                messages = messages[1:]

            # First user message (often contains important context)
            first_user = None
            remaining_msgs = []
            for msg in messages:
                role = msg.role if hasattr(msg, 'role') else msg.get('role', '')
                if role == 'user' and first_user is None:
                    first_user = msg
                else:
                    remaining_msgs.append(msg)

            if first_user:
                result.append(first_user)

            # Fill with recent messages
            remaining = available - sum(self._message_tokens(m) for m in result)
            selected = []

            for msg in reversed(remaining_msgs):
                tokens = self._message_tokens(msg)
                if remaining >= tokens:
                    selected.insert(0, msg)
                    remaining -= tokens

            result.extend(selected)
            messages = result

        tokens_used = sum(self._message_tokens(m) for m in messages)
        self.set_output_val(0, Data(messages))
        self.set_output_val(1, Data(tokens_used))
        self.set_output_val(2, Data(truncated))
        self.exec_output(3)


class ContextCompressorNode(Node):
    """
    Compress context to fit more information.

    Uses summarization or extraction to reduce
    context size while preserving key information.

    Inputs:
        - context: Text to compress
        - target_ratio: Target compression ratio (0-1)
        - method: 'extractive', 'keywords', 'first_last'
        - exec: Trigger

    Outputs:
        - compressed: Compressed context
        - original_length: Original character count
        - compressed_length: Compressed character count
        - done: Exec
    """

    title = 'Context Compressor'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='context'),
        NodeInputType(label='target_ratio', default=0.5),
        NodeInputType(label='method', default='extractive'),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='compressed'),
        NodeOutputType(label='original_length'),
        NodeOutputType(label='compressed_length'),
        NodeOutputType(type_='exec', label='done'),
    ]

    def _extractive_compress(self, text: str, ratio: float) -> str:
        """Extract key sentences based on position and keywords."""
        import re

        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) <= 2:
            return text

        target_count = max(1, int(len(sentences) * ratio))

        # Score sentences by position and keyword presence
        keywords = set(['important', 'key', 'main', 'summary', 'conclusion',
                       'result', 'finding', 'note', 'critical'])

        scored = []
        for i, sent in enumerate(sentences):
            score = 0

            # Position scoring (first and last sentences important)
            if i == 0:
                score += 3
            elif i == len(sentences) - 1:
                score += 2
            elif i < 3:
                score += 1

            # Keyword scoring
            words = set(sent.lower().split())
            score += len(words & keywords)

            scored.append((score, i, sent))

        # Select top sentences
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = scored[:target_count]

        # Restore original order
        selected.sort(key=lambda x: x[1])

        return ' '.join(s[2] for s in selected)

    def _keyword_compress(self, text: str, ratio: float) -> str:
        """Extract just keywords and key phrases."""
        import re

        # Extract potential key phrases (capitalized, quoted, etc.)
        patterns = [
            r'"[^"]*"',  # Quoted text
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',  # Proper nouns
            r'\b\d+(?:\.\d+)?%?\b',  # Numbers
        ]

        extracted = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            extracted.extend(matches)

        # Also get first sentence
        first_sent = re.split(r'(?<=[.!?])\s+', text)[0] if text else ""

        result = first_sent + "\n\nKey points: " + ", ".join(set(extracted)[:10])
        return result

    def _first_last_compress(self, text: str, ratio: float) -> str:
        """Keep first and last portions."""
        target_len = int(len(text) * ratio)
        half = target_len // 2

        return text[:half] + "\n...\n" + text[-half:]

    def update_event(self, inp=-1):
        if inp != 3:
            return

        context_input = self.input(0)
        if not context_input or not context_input.payload:
            return

        text = str(context_input.payload)
        original_len = len(text)

        ratio = 0.5
        ratio_input = self.input(1)
        if ratio_input and ratio_input.payload is not None:
            ratio = float(ratio_input.payload)

        method = 'extractive'
        method_input = self.input(2)
        if method_input and method_input.payload:
            method = str(method_input.payload).lower()

        if method == 'keywords':
            compressed = self._keyword_compress(text, ratio)
        elif method == 'first_last':
            compressed = self._first_last_compress(text, ratio)
        else:
            compressed = self._extractive_compress(text, ratio)

        self.set_output_val(0, Data(compressed))
        self.set_output_val(1, Data(original_len))
        self.set_output_val(2, Data(len(compressed)))
        self.exec_output(3)


class RelevanceFilterNode(Node):
    """
    Filter context by relevance to query.

    Removes irrelevant sections to focus context
    on the most pertinent information.

    Inputs:
        - context: Context to filter (text or chunks)
        - query: Query to check relevance against
        - threshold: Minimum relevance score (0-1)
        - max_chunks: Maximum chunks to keep
        - exec: Trigger

    Outputs:
        - relevant: Relevant context
        - removed_count: Number of chunks removed
        - done: Exec
    """

    title = 'Relevance Filter'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='context'),
        NodeInputType(label='query'),
        NodeInputType(label='threshold', default=0.3),
        NodeInputType(label='max_chunks', default=10),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='relevant'),
        NodeOutputType(label='removed_count'),
        NodeOutputType(type_='exec', label='done'),
    ]

    def _simple_relevance(self, text: str, query: str) -> float:
        """Simple keyword overlap relevance score."""
        text_words = set(text.lower().split())
        query_words = set(query.lower().split())

        if not query_words:
            return 0.5

        overlap = len(text_words & query_words)
        return overlap / len(query_words)

    def update_event(self, inp=-1):
        if inp != 4:
            return

        context_input = self.input(0)
        if not context_input or not context_input.payload:
            return

        context = context_input.payload

        query = ""
        query_input = self.input(1)
        if query_input and query_input.payload:
            query = str(query_input.payload)

        threshold = 0.3
        th_input = self.input(2)
        if th_input and th_input.payload is not None:
            threshold = float(th_input.payload)

        max_chunks = 10
        mc_input = self.input(3)
        if mc_input and mc_input.payload:
            max_chunks = int(mc_input.payload)

        # Handle both string and list input
        if isinstance(context, str):
            chunks = context.split('\n\n')
        elif isinstance(context, list):
            chunks = [str(c) for c in context]
        else:
            chunks = [str(context)]

        original_count = len(chunks)

        # Score and filter chunks
        scored = [
            (chunk, self._simple_relevance(chunk, query))
            for chunk in chunks
        ]

        # Filter by threshold
        relevant = [(c, s) for c, s in scored if s >= threshold]

        # Sort by relevance and limit
        relevant.sort(key=lambda x: x[1], reverse=True)
        relevant = relevant[:max_chunks]

        # Extract just the chunks
        result_chunks = [c for c, s in relevant]

        if isinstance(context, str):
            result = '\n\n'.join(result_chunks)
        else:
            result = result_chunks

        removed = original_count - len(result_chunks)

        self.set_output_val(0, Data(result))
        self.set_output_val(1, Data(removed))
        self.exec_output(2)


# Export context nodes
context_nodes = [
    ContextWindowNode,
    ContextCompressorNode,
    RelevanceFilterNode,
]
