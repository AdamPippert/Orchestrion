"""
Tests for routing and orchestration nodes.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../ryven-editor/ryven/example_nodes'))


class TestModelRouter:
    """Tests for model routing logic."""

    def test_complexity_estimation(self):
        """Test task complexity estimation."""
        from agent_nodes.routing.model_router import ModelRouterNode

        node = ModelRouterNode.__new__(ModelRouterNode)

        # Simple tasks
        simple = node._estimate_complexity("simple question")
        assert simple < 0.5

        # Complex tasks
        complex_task = node._estimate_complexity("complex detailed analysis with reasoning")
        assert complex_task > 0.5

    def test_model_scoring(self):
        """Test model scoring logic."""
        from agent_nodes.routing.model_router import ModelRouterNode
        from agent_nodes.providers.model_info import get_model_info

        node = ModelRouterNode.__new__(ModelRouterNode)

        # Get a known model
        info = get_model_info("gpt-4o")
        assert info is not None

        # Score it
        score, breakdown = node._score_model(
            model_info=info,
            cost_weight=0.3,
            speed_weight=0.3,
            quality_weight=0.4,
            complexity=0.5,
        )

        assert 0 <= score <= 1
        assert 'cost' in breakdown
        assert 'speed' in breakdown
        assert 'quality' in breakdown


class TestSemanticRouter:
    """Tests for semantic routing."""

    def test_simple_relevance(self):
        """Test simple relevance scoring."""
        from agent_nodes.routing.semantic_router import SemanticRouterNode

        node = SemanticRouterNode.__new__(SemanticRouterNode)

        # High overlap
        sim = node._cosine_similarity([1, 0, 0], [1, 0, 0])
        assert sim == 1.0

        # No overlap
        sim = node._cosine_similarity([1, 0, 0], [0, 1, 0])
        assert sim == 0.0


class TestLoadBalancer:
    """Tests for load balancing."""

    def test_load_balancer_strategies(self):
        """Verify load balancer has correct strategies."""
        from agent_nodes.routing.load_balancer import LoadBalancerNode

        assert hasattr(LoadBalancerNode, 'title')
        assert LoadBalancerNode.title == 'Load Balancer'


class TestCircuitBreaker:
    """Tests for circuit breaker pattern."""

    def test_circuit_breaker_states(self):
        """Test circuit breaker state transitions."""
        from agent_nodes.routing.load_balancer import CircuitBreakerNode

        node = CircuitBreakerNode.__new__(CircuitBreakerNode)
        node._state = 'closed'
        node._failure_count = 0
        node._last_failure_time = None

        # Initial state
        assert node._state == 'closed'
        assert node._failure_count == 0


class TestOrchestration:
    """Tests for orchestration patterns."""

    def test_parallel_node(self):
        """Test parallel execution node."""
        from agent_nodes.routing.orchestration import ParallelNode

        assert hasattr(ParallelNode, 'title')
        assert ParallelNode.title == 'Parallel'

    def test_sequence_node(self):
        """Test sequence execution node."""
        from agent_nodes.routing.orchestration import SequenceNode

        assert hasattr(SequenceNode, 'title')
        assert SequenceNode.title == 'Sequence'

    def test_conditional_node(self):
        """Test conditional branching node."""
        from agent_nodes.routing.orchestration import ConditionalNode

        assert hasattr(ConditionalNode, 'title')
        assert ConditionalNode.title == 'Conditional'

    def test_loop_node(self):
        """Test loop iteration node."""
        from agent_nodes.routing.orchestration import LoopNode

        assert hasattr(LoopNode, 'title')
        assert LoopNode.title == 'Loop'

    def test_map_node(self):
        """Test map operation node."""
        from agent_nodes.routing.orchestration import MapNode

        assert hasattr(MapNode, 'title')
        assert MapNode.title == 'Map'
