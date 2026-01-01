"""
Semantic routing nodes.

Route requests based on content semantics, intent,
or topic classification.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from ryven.node_env import *


class SemanticRouterNode(Node):
    """
    Route based on semantic similarity to route definitions.

    Uses embeddings to match input to predefined routes
    based on example descriptions.

    Inputs:
        - input: Text to route
        - routes: Dict of route_name -> example descriptions
        - embedder: Provider for generating embeddings
        - threshold: Minimum similarity to match (0-1)
        - exec: Trigger

    Outputs:
        - route: Matched route name
        - confidence: Confidence score
        - all_scores: Scores for all routes
        - matched: Exec if route matched
        - no_match: Exec if no route matched
    """

    title = 'Semantic Router'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='input'),
        NodeInputType(label='routes'),
        NodeInputType(label='embedder'),
        NodeInputType(label='threshold', default=0.7),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='route'),
        NodeOutputType(label='confidence'),
        NodeOutputType(label='all_scores'),
        NodeOutputType(type_='exec', label='matched'),
        NodeOutputType(type_='exec', label='no_match'),
    ]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def update_event(self, inp=-1):
        if inp != 4:
            return

        input_val = self.input(0)
        if not input_val or not input_val.payload:
            return

        input_text = str(input_val.payload)

        routes_input = self.input(1)
        if not routes_input or not routes_input.payload:
            return

        routes = routes_input.payload
        if not isinstance(routes, dict):
            return

        # For now, use simple keyword matching
        # In a full implementation, this would use embeddings
        scores = []
        for route_name, examples in routes.items():
            if isinstance(examples, str):
                examples = [examples]

            max_score = 0.0
            for example in examples:
                # Simple keyword overlap score
                input_words = set(input_text.lower().split())
                example_words = set(example.lower().split())
                if len(input_words) > 0 and len(example_words) > 0:
                    overlap = len(input_words & example_words)
                    score = overlap / max(len(input_words), len(example_words))
                    max_score = max(max_score, score)

            scores.append((route_name, max_score))

        threshold = 0.7
        thresh_input = self.input(3)
        if thresh_input and thresh_input.payload is not None:
            threshold = float(thresh_input.payload)

        scores.sort(key=lambda x: x[1], reverse=True)
        all_scores = [{'route': r, 'score': s} for r, s in scores]

        if scores and scores[0][1] >= threshold:
            best_route, confidence = scores[0]
            self.set_output_val(0, Data(best_route))
            self.set_output_val(1, Data(confidence))
            self.set_output_val(2, Data(all_scores))
            self.exec_output(3)
        else:
            self.set_output_val(0, Data(None))
            self.set_output_val(1, Data(0.0))
            self.set_output_val(2, Data(all_scores))
            self.exec_output(4)


class IntentClassifierNode(Node):
    """
    Classify user intent from input text.

    Determines what the user is trying to do based on
    their input, routing to appropriate handlers.

    Inputs:
        - input: User input text
        - intents: List of intent definitions
        - provider: LLM provider for classification
        - exec: Trigger

    Outputs:
        - intent: Detected intent name
        - confidence: Confidence score
        - entities: Extracted entities from input
        - done: Exec
    """

    title = 'Intent Classifier'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='input'),
        NodeInputType(label='intents'),
        NodeInputType(label='provider'),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='intent'),
        NodeOutputType(label='confidence'),
        NodeOutputType(label='entities'),
        NodeOutputType(type_='exec', label='done'),
    ]

    # Common intent patterns
    INTENT_PATTERNS = {
        'greeting': ['hello', 'hi', 'hey', 'good morning', 'good afternoon'],
        'farewell': ['bye', 'goodbye', 'see you', 'later'],
        'help': ['help', 'assist', 'support', 'how do i', 'can you help'],
        'question': ['what', 'why', 'how', 'when', 'where', 'who', '?'],
        'command': ['do', 'make', 'create', 'delete', 'update', 'run'],
        'search': ['find', 'search', 'look for', 'show me'],
        'feedback': ['feedback', 'suggest', 'improve', 'bug', 'issue'],
    }

    def update_event(self, inp=-1):
        if inp != 3:
            return

        input_val = self.input(0)
        if not input_val or not input_val.payload:
            return

        input_text = str(input_val.payload).lower()

        # Get custom intents or use defaults
        intents = dict(self.INTENT_PATTERNS)
        intents_input = self.input(1)
        if intents_input and intents_input.payload:
            custom = intents_input.payload
            if isinstance(custom, dict):
                intents.update(custom)

        # Score each intent
        scores = []
        for intent, patterns in intents.items():
            score = 0.0
            for pattern in patterns:
                if pattern in input_text:
                    score = max(score, len(pattern) / len(input_text))

            if score > 0:
                scores.append((intent, min(score * 2, 1.0)))  # Scale up

        if scores:
            scores.sort(key=lambda x: x[1], reverse=True)
            intent, confidence = scores[0]
        else:
            intent = 'unknown'
            confidence = 0.0

        # Simple entity extraction (in full impl, use NER)
        entities = {}

        self.set_output_val(0, Data(intent))
        self.set_output_val(1, Data(confidence))
        self.set_output_val(2, Data(entities))
        self.exec_output(3)


class TopicRouterNode(Node):
    """
    Route based on topic detection.

    Detects the topic of input and routes to
    topic-specific handlers.

    Inputs:
        - input: Text to classify
        - topics: Dict of topic -> keywords
        - default_topic: Fallback topic
        - exec: Trigger

    Outputs:
        - topic: Detected topic
        - confidence: Match confidence
        - route_<topic>: Dynamic exec outputs for each topic
    """

    title = 'Topic Router'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='input'),
        NodeInputType(label='topics'),
        NodeInputType(label='default_topic'),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='topic'),
        NodeOutputType(label='confidence'),
        NodeOutputType(type_='exec', label='technical'),
        NodeOutputType(type_='exec', label='business'),
        NodeOutputType(type_='exec', label='support'),
        NodeOutputType(type_='exec', label='other'),
    ]

    DEFAULT_TOPICS = {
        'technical': ['code', 'api', 'bug', 'error', 'debug', 'programming', 'database'],
        'business': ['sales', 'revenue', 'customer', 'market', 'strategy', 'pricing'],
        'support': ['help', 'issue', 'problem', 'not working', 'broken', 'fix'],
        'other': [],
    }

    def update_event(self, inp=-1):
        if inp != 3:
            return

        input_val = self.input(0)
        if not input_val or not input_val.payload:
            return

        input_text = str(input_val.payload).lower()

        topics = dict(self.DEFAULT_TOPICS)
        topics_input = self.input(1)
        if topics_input and topics_input.payload:
            if isinstance(topics_input.payload, dict):
                topics.update(topics_input.payload)

        default = 'other'
        default_input = self.input(2)
        if default_input and default_input.payload:
            default = str(default_input.payload)

        # Score topics
        scores = []
        for topic, keywords in topics.items():
            score = sum(1 for kw in keywords if kw in input_text)
            if score > 0:
                scores.append((topic, score / len(keywords) if keywords else 0))

        if scores:
            scores.sort(key=lambda x: x[1], reverse=True)
            topic, confidence = scores[0]
        else:
            topic = default
            confidence = 0.0

        self.set_output_val(0, Data(topic))
        self.set_output_val(1, Data(confidence))

        # Trigger appropriate exec output
        topic_to_output = {
            'technical': 2,
            'business': 3,
            'support': 4,
            'other': 5,
        }
        output_idx = topic_to_output.get(topic, 5)
        self.exec_output(output_idx)


# Export semantic router nodes
semantic_router_nodes = [
    SemanticRouterNode,
    IntentClassifierNode,
    TopicRouterNode,
]
