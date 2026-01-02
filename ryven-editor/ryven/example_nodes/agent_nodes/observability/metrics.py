"""
Metrics collection and monitoring nodes.
"""

from __future__ import annotations
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict
from ryven.node_env import *


class TokenCounterNode(Node):
    """
    Count and track token usage.

    Monitors token consumption across the workflow
    for optimization and cost control.

    Inputs:
        - text: Text to count tokens for
        - response: LLM response to extract tokens from
        - model: Model name for accurate counting
        - exec: Trigger

    Outputs:
        - tokens: Token count for this input
        - session_total: Total tokens this session
        - breakdown: Breakdown by model
        - done: Exec
    """

    title = 'Token Counter'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='text'),
        NodeInputType(label='response'),
        NodeInputType(label='model'),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='tokens'),
        NodeOutputType(label='session_total'),
        NodeOutputType(label='breakdown'),
        NodeOutputType(type_='exec', label='done'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self._session_total = 0
        self._by_model: Dict[str, int] = defaultdict(int)

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (4 chars per token for English)."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def update_event(self, inp=-1):
        if inp != 3:
            return

        tokens = 0
        model = "unknown"

        model_input = self.input(2)
        if model_input and model_input.payload:
            model = str(model_input.payload)

        # Check for response with token info
        response_input = self.input(1)
        if response_input and response_input.payload:
            response = response_input.payload
            if hasattr(response, 'total_tokens'):
                tokens = response.total_tokens
            elif isinstance(response, dict):
                tokens = response.get('total_tokens', 0)

        # Fall back to text estimation
        if tokens == 0:
            text_input = self.input(0)
            if text_input and text_input.payload:
                text = str(text_input.payload)
                tokens = self._estimate_tokens(text)

        self._session_total += tokens
        self._by_model[model] += tokens

        self.set_output_val(0, Data(tokens))
        self.set_output_val(1, Data(self._session_total))
        self.set_output_val(2, Data(dict(self._by_model)))
        self.exec_output(3)

    def get_state(self) -> dict:
        return {
            'session_total': self._session_total,
            'by_model': dict(self._by_model),
        }

    def set_state(self, data: dict, version):
        self._session_total = data.get('session_total', 0)
        self._by_model = defaultdict(int, data.get('by_model', {}))


class CostTrackerNode(Node):
    """
    Track costs across LLM operations.

    Calculates and aggregates costs based on token
    usage and model pricing.

    Inputs:
        - response: LLM response with token info
        - model: Model name for pricing
        - custom_pricing: Override default pricing
        - exec: Trigger

    Outputs:
        - request_cost: Cost for this request
        - session_cost: Total session cost
        - by_model: Cost breakdown by model
        - by_hour: Cost breakdown by hour
        - done: Exec
    """

    title = 'Cost Tracker'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='response'),
        NodeInputType(label='model'),
        NodeInputType(label='custom_pricing'),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='request_cost'),
        NodeOutputType(label='session_cost'),
        NodeOutputType(label='by_model'),
        NodeOutputType(label='by_hour'),
        NodeOutputType(type_='exec', label='done'),
    ]

    # Default pricing per 1M tokens
    DEFAULT_PRICING = {
        'gpt-4o': {'input': 2.50, 'output': 10.00},
        'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
        'o1': {'input': 15.00, 'output': 60.00},
        'o1-mini': {'input': 3.00, 'output': 12.00},
        'claude-sonnet-4-20250514': {'input': 3.00, 'output': 15.00},
        'claude-opus-4-20250514': {'input': 15.00, 'output': 75.00},
        'claude-3-5-haiku-20241022': {'input': 0.80, 'output': 4.00},
    }

    def __init__(self, params):
        super().__init__(params)
        self._session_cost = 0.0
        self._by_model: Dict[str, float] = defaultdict(float)
        self._by_hour: Dict[str, float] = defaultdict(float)

    def _calculate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
        custom_pricing: Optional[Dict] = None,
    ) -> float:
        pricing = custom_pricing or self.DEFAULT_PRICING.get(model, {'input': 1.0, 'output': 2.0})

        input_cost = (prompt_tokens / 1_000_000) * pricing['input']
        output_cost = (completion_tokens / 1_000_000) * pricing['output']

        return input_cost + output_cost

    def update_event(self, inp=-1):
        if inp != 3:
            return

        response_input = self.input(0)
        if not response_input or not response_input.payload:
            return

        response = response_input.payload

        prompt_tokens = 0
        completion_tokens = 0
        model = "unknown"

        if hasattr(response, 'prompt_tokens'):
            prompt_tokens = response.prompt_tokens
            completion_tokens = response.completion_tokens
            model = response.model
        elif isinstance(response, dict):
            prompt_tokens = response.get('prompt_tokens', 0)
            completion_tokens = response.get('completion_tokens', 0)
            model = response.get('model', 'unknown')

        model_input = self.input(1)
        if model_input and model_input.payload:
            model = str(model_input.payload)

        custom_pricing = None
        pricing_input = self.input(2)
        if pricing_input and pricing_input.payload:
            custom_pricing = pricing_input.payload

        cost = self._calculate_cost(prompt_tokens, completion_tokens, model, custom_pricing)

        self._session_cost += cost
        self._by_model[model] += cost

        hour_key = datetime.now().strftime('%Y-%m-%d %H:00')
        self._by_hour[hour_key] += cost

        self.set_output_val(0, Data(cost))
        self.set_output_val(1, Data(self._session_cost))
        self.set_output_val(2, Data(dict(self._by_model)))
        self.set_output_val(3, Data(dict(self._by_hour)))
        self.exec_output(4)

    def get_state(self) -> dict:
        return {
            'session_cost': self._session_cost,
            'by_model': dict(self._by_model),
            'by_hour': dict(self._by_hour),
        }

    def set_state(self, data: dict, version):
        self._session_cost = data.get('session_cost', 0.0)
        self._by_model = defaultdict(float, data.get('by_model', {}))
        self._by_hour = defaultdict(float, data.get('by_hour', {}))


class LatencyMonitorNode(Node):
    """
    Monitor and track latency of operations.

    Measures time-to-first-token and total latency
    for performance optimization.

    Inputs:
        - response: Response with latency info
        - operation: Name of operation
        - start_time: Manual start time (optional)
        - exec: Trigger

    Outputs:
        - latency_ms: Latency for this operation
        - avg_latency: Average latency
        - p95_latency: 95th percentile latency
        - by_operation: Breakdown by operation
        - done: Exec
    """

    title = 'Latency Monitor'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='response'),
        NodeInputType(label='operation'),
        NodeInputType(label='start_time'),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='latency_ms'),
        NodeOutputType(label='avg_latency'),
        NodeOutputType(label='p95_latency'),
        NodeOutputType(label='by_operation'),
        NodeOutputType(type_='exec', label='done'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self._latencies: List[float] = []
        self._by_operation: Dict[str, List[float]] = defaultdict(list)

    def _percentile(self, values: List[float], p: float) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * p / 100)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    def update_event(self, inp=-1):
        if inp != 3:
            return

        latency_ms = 0.0
        operation = "default"

        op_input = self.input(1)
        if op_input and op_input.payload:
            operation = str(op_input.payload)

        # Check response for latency
        response_input = self.input(0)
        if response_input and response_input.payload:
            response = response_input.payload
            if hasattr(response, 'latency_ms'):
                latency_ms = response.latency_ms
            elif isinstance(response, dict):
                latency_ms = response.get('latency_ms', 0)

        # Check manual start time
        if latency_ms == 0:
            start_input = self.input(2)
            if start_input and start_input.payload:
                start_time = float(start_input.payload)
                latency_ms = (time.time() - start_time) * 1000

        self._latencies.append(latency_ms)
        self._by_operation[operation].append(latency_ms)

        avg = sum(self._latencies) / len(self._latencies) if self._latencies else 0
        p95 = self._percentile(self._latencies, 95)

        by_op_stats = {}
        for op, lats in self._by_operation.items():
            by_op_stats[op] = {
                'avg': sum(lats) / len(lats) if lats else 0,
                'p95': self._percentile(lats, 95),
                'count': len(lats),
            }

        self.set_output_val(0, Data(latency_ms))
        self.set_output_val(1, Data(avg))
        self.set_output_val(2, Data(p95))
        self.set_output_val(3, Data(by_op_stats))
        self.exec_output(4)

    def get_state(self) -> dict:
        return {
            'latencies': self._latencies,
            'by_operation': dict(self._by_operation),
        }

    def set_state(self, data: dict, version):
        self._latencies = data.get('latencies', [])
        self._by_operation = defaultdict(list, data.get('by_operation', {}))


class MetricsAggregatorNode(Node):
    """
    Aggregate and summarize all metrics.

    Combines token, cost, and latency metrics
    into a unified dashboard view.

    Inputs:
        - token_metrics: From TokenCounterNode
        - cost_metrics: From CostTrackerNode
        - latency_metrics: From LatencyMonitorNode
        - exec: Trigger refresh

    Outputs:
        - summary: Combined metrics summary
        - health_status: Overall health status
        - alerts: Any metric alerts
        - done: Exec
    """

    title = 'Metrics Aggregator'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='token_metrics'),
        NodeInputType(label='cost_metrics'),
        NodeInputType(label='latency_metrics'),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='summary'),
        NodeOutputType(label='health_status'),
        NodeOutputType(label='alerts'),
        NodeOutputType(type_='exec', label='done'),
    ]

    def update_event(self, inp=-1):
        if inp != 3:
            return

        token_input = self.input(0)
        cost_input = self.input(1)
        latency_input = self.input(2)

        summary = {
            'timestamp': datetime.now().isoformat(),
            'tokens': {},
            'costs': {},
            'latency': {},
        }

        alerts = []
        health = 'healthy'

        if token_input and token_input.payload:
            summary['tokens'] = token_input.payload

        if cost_input and cost_input.payload:
            summary['costs'] = cost_input.payload
            # Alert on high costs
            session_cost = cost_input.payload.get('session_cost', 0)
            if session_cost > 10.0:
                alerts.append({
                    'type': 'cost_warning',
                    'message': f'High session cost: ${session_cost:.2f}',
                    'severity': 'warning',
                })
                health = 'warning'

        if latency_input and latency_input.payload:
            summary['latency'] = latency_input.payload
            # Alert on high latency
            p95 = latency_input.payload.get('p95_latency', 0)
            if p95 > 5000:  # 5 seconds
                alerts.append({
                    'type': 'latency_warning',
                    'message': f'High P95 latency: {p95:.0f}ms',
                    'severity': 'warning',
                })
                health = 'warning'

        self.set_output_val(0, Data(summary))
        self.set_output_val(1, Data(health))
        self.set_output_val(2, Data(alerts))
        self.exec_output(3)


# Export metrics nodes
metrics_nodes = [
    TokenCounterNode,
    CostTrackerNode,
    LatencyMonitorNode,
    MetricsAggregatorNode,
]
