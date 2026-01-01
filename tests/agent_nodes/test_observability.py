"""
Tests for observability nodes.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../ryven-editor/ryven/example_nodes'))


class TestTokenCounter:
    """Tests for token counting."""

    def test_token_estimation(self):
        """Test token estimation."""
        from agent_nodes.observability.metrics import TokenCounterNode

        node = TokenCounterNode.__new__(TokenCounterNode)

        tokens = node._estimate_tokens("Hello world!")
        assert tokens == 3  # 12 chars / 4 = 3


class TestCostTracker:
    """Tests for cost tracking."""

    def test_cost_calculation(self):
        """Test cost calculation with known pricing."""
        from agent_nodes.observability.metrics import CostTrackerNode

        node = CostTrackerNode.__new__(CostTrackerNode)

        # Test with gpt-4o pricing
        cost = node._calculate_cost(
            prompt_tokens=1000000,  # 1M tokens
            completion_tokens=500000,  # 500K tokens
            model="gpt-4o",
        )

        # gpt-4o: $2.50/1M input + $10/1M output
        expected = 2.50 + 5.00  # $7.50 total
        assert abs(cost - expected) < 0.01


class TestLatencyMonitor:
    """Tests for latency monitoring."""

    def test_percentile_calculation(self):
        """Test percentile calculation."""
        from agent_nodes.observability.metrics import LatencyMonitorNode

        node = LatencyMonitorNode.__new__(LatencyMonitorNode)

        values = list(range(1, 101))  # 1 to 100

        p50 = node._percentile(values, 50)
        assert p50 == 50

        p95 = node._percentile(values, 95)
        assert p95 == 95

    def test_empty_percentile(self):
        """Test percentile with empty list."""
        from agent_nodes.observability.metrics import LatencyMonitorNode

        node = LatencyMonitorNode.__new__(LatencyMonitorNode)

        result = node._percentile([], 95)
        assert result == 0.0


class TestMetricsAggregator:
    """Tests for metrics aggregation."""

    def test_aggregator_node_exists(self):
        """Verify MetricsAggregatorNode is properly defined."""
        from agent_nodes.observability.metrics import MetricsAggregatorNode

        assert hasattr(MetricsAggregatorNode, 'title')
        assert MetricsAggregatorNode.title == 'Metrics Aggregator'


class TestTracing:
    """Tests for tracing functionality."""

    def test_trace_node(self):
        """Test TraceNode class."""
        from agent_nodes.observability.tracing import TraceNode

        assert hasattr(TraceNode, 'title')
        assert TraceNode.title == 'Trace'
        assert hasattr(TraceNode, 'get_current_trace_id')
        assert hasattr(TraceNode, 'add_span')

    def test_span_node(self):
        """Test SpanNode class."""
        from agent_nodes.observability.tracing import SpanNode

        assert hasattr(SpanNode, 'title')
        assert SpanNode.title == 'Span'

    def test_trace_viewer(self):
        """Test TraceViewerNode class."""
        from agent_nodes.observability.tracing import TraceViewerNode

        assert hasattr(TraceViewerNode, 'title')
        assert TraceViewerNode.title == 'Trace Viewer'


class TestDebugging:
    """Tests for debugging tools."""

    def test_debug_formatting(self):
        """Test debug data formatting."""
        from agent_nodes.observability.debugging import DebugNode

        node = DebugNode.__new__(DebugNode)

        # String formatting
        formatted = node._format_data("test", verbose=False)
        assert '"test"' in formatted

        # Long string truncation
        long_str = "A" * 500
        formatted = node._format_data(long_str, verbose=False)
        assert "..." in formatted
        assert "500 chars" in formatted

        # List formatting
        formatted = node._format_data([1, 2, 3], verbose=False)
        assert "3 items" in formatted

        # Dict formatting
        formatted = node._format_data({"a": 1, "b": 2}, verbose=False)
        assert "keys" in formatted

    def test_inspector_size(self):
        """Test inspector size calculation."""
        from agent_nodes.observability.debugging import InspectorNode

        node = InspectorNode.__new__(InspectorNode)

        # String size
        size = node._get_size("hello")
        assert size['chars'] == 5
        assert size['estimated_tokens'] == 1

        # List size
        size = node._get_size([1, 2, 3])
        assert size['items'] == 3

        # Dict size
        size = node._get_size({"a": 1, "b": 2})
        assert size['keys'] == 2

    def test_schema_inference(self):
        """Test schema inference."""
        from agent_nodes.observability.debugging import InspectorNode

        node = InspectorNode.__new__(InspectorNode)

        # Simple types
        schema = node._infer_schema("hello")
        assert schema['type'] == 'string'

        schema = node._infer_schema(42)
        assert schema['type'] == 'integer'

        schema = node._infer_schema(3.14)
        assert schema['type'] == 'number'

        schema = node._infer_schema(True)
        assert schema['type'] == 'boolean'

        schema = node._infer_schema(None)
        assert schema['type'] == 'null'

        # Array
        schema = node._infer_schema([1, 2, 3])
        assert schema['type'] == 'array'
        assert schema['length'] == 3

        # Object
        schema = node._infer_schema({"name": "test", "value": 42})
        assert schema['type'] == 'object'
        assert 'name' in schema['properties']
        assert 'value' in schema['properties']

    def test_breakpoint_node(self):
        """Test BreakpointNode class."""
        from agent_nodes.observability.debugging import BreakpointNode

        assert hasattr(BreakpointNode, 'title')
        assert BreakpointNode.title == 'Breakpoint'
