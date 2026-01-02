"""
Load balancing and fallback nodes.

Distribute requests across providers and handle failures.
"""

from __future__ import annotations
import random
import time
from typing import List, Dict, Any, Optional
from collections import deque
from ryven.node_env import *


class LoadBalancerNode(Node):
    """
    Distribute requests across multiple providers.

    Supports multiple balancing strategies:
    - round_robin: Rotate through providers
    - random: Random selection
    - weighted: Weight by cost/performance
    - least_loaded: Track and balance load

    Inputs:
        - providers: List of available providers
        - strategy: Balancing strategy
        - weights: Optional weights for weighted strategy
        - exec: Trigger

    Outputs:
        - provider: Selected provider
        - index: Index of selected provider
        - stats: Load balancer statistics
        - done: Exec
    """

    title = 'Load Balancer'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='providers'),
        NodeInputType(label='strategy', default='round_robin'),
        NodeInputType(label='weights'),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='provider'),
        NodeOutputType(label='index'),
        NodeOutputType(label='stats'),
        NodeOutputType(type_='exec', label='done'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self._rr_index = 0
        self._load_counts: Dict[int, int] = {}
        self._request_count = 0

    def update_event(self, inp=-1):
        if inp != 3:
            return

        providers_input = self.input(0)
        if not providers_input or not providers_input.payload:
            return

        providers = providers_input.payload
        if not isinstance(providers, list):
            providers = [providers]

        if not providers:
            return

        strategy = 'round_robin'
        strat_input = self.input(1)
        if strat_input and strat_input.payload:
            strategy = str(strat_input.payload).lower()

        weights = None
        weights_input = self.input(2)
        if weights_input and weights_input.payload:
            weights = weights_input.payload

        selected_idx = 0

        if strategy == 'round_robin':
            selected_idx = self._rr_index % len(providers)
            self._rr_index += 1

        elif strategy == 'random':
            selected_idx = random.randint(0, len(providers) - 1)

        elif strategy == 'weighted' and weights:
            if isinstance(weights, list) and len(weights) == len(providers):
                total = sum(weights)
                r = random.random() * total
                cumulative = 0
                for i, w in enumerate(weights):
                    cumulative += w
                    if r <= cumulative:
                        selected_idx = i
                        break

        elif strategy == 'least_loaded':
            min_load = float('inf')
            for i in range(len(providers)):
                load = self._load_counts.get(i, 0)
                if load < min_load:
                    min_load = load
                    selected_idx = i
            self._load_counts[selected_idx] = self._load_counts.get(selected_idx, 0) + 1

        self._request_count += 1

        stats = {
            'total_requests': self._request_count,
            'strategy': strategy,
            'provider_loads': dict(self._load_counts),
        }

        self.set_output_val(0, Data(providers[selected_idx]))
        self.set_output_val(1, Data(selected_idx))
        self.set_output_val(2, Data(stats))
        self.exec_output(3)

    def get_state(self) -> dict:
        return {
            'rr_index': self._rr_index,
            'load_counts': self._load_counts,
            'request_count': self._request_count,
        }

    def set_state(self, data: dict, version):
        self._rr_index = data.get('rr_index', 0)
        self._load_counts = data.get('load_counts', {})
        self._request_count = data.get('request_count', 0)


class FallbackChainNode(Node):
    """
    Try providers in order until one succeeds.

    Implements cascading fallback for reliability.

    Inputs:
        - providers: Ordered list of providers to try
        - operation: Function/node to execute
        - input_data: Data to pass to operation
        - max_retries: Retries per provider before fallback
        - exec: Trigger

    Outputs:
        - result: Result from successful provider
        - provider_used: Which provider succeeded
        - attempts: Number of attempts made
        - success: Exec on success
        - all_failed: Exec if all providers failed
    """

    title = 'Fallback Chain'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='providers'),
        NodeInputType(label='input_data'),
        NodeInputType(label='max_retries', default=1),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='result'),
        NodeOutputType(label='provider_used'),
        NodeOutputType(label='attempts'),
        NodeOutputType(type_='exec', label='success'),
        NodeOutputType(type_='exec', label='all_failed'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self._current_provider_idx = 0
        self._total_attempts = 0

    def update_event(self, inp=-1):
        if inp != 3:
            return

        providers_input = self.input(0)
        if not providers_input or not providers_input.payload:
            return

        providers = providers_input.payload
        if not isinstance(providers, list):
            providers = [providers]

        max_retries = 1
        retry_input = self.input(2)
        if retry_input and retry_input.payload is not None:
            max_retries = int(retry_input.payload)

        # Note: In a real implementation, this would execute
        # the actual operation. Here we just select the provider.
        # The actual execution would happen in connected nodes.

        self._total_attempts = 0
        provider_used = None

        for idx, provider in enumerate(providers):
            for attempt in range(max_retries):
                self._total_attempts += 1
                # In real impl, try operation here
                # For now, assume first provider works
                provider_used = provider
                break
            if provider_used:
                break

        if provider_used:
            self.set_output_val(0, Data(None))  # Result would come from operation
            self.set_output_val(1, Data(provider_used))
            self.set_output_val(2, Data(self._total_attempts))
            self.exec_output(3)
        else:
            self.set_output_val(2, Data(self._total_attempts))
            self.exec_output(4)


class RetryNode(Node):
    """
    Retry an operation with exponential backoff.

    Wraps operations with automatic retry logic.

    Inputs:
        - max_retries: Maximum number of retry attempts
        - initial_delay: Initial delay between retries (seconds)
        - max_delay: Maximum delay between retries
        - backoff_factor: Multiplier for delay after each retry
        - exec_try: Trigger attempt
        - exec_success: Signal operation succeeded
        - exec_failure: Signal operation failed

    Outputs:
        - attempt: Current attempt number
        - delay: Current delay before next retry
        - retry: Exec to trigger retry
        - exhausted: Exec when retries exhausted
        - succeeded: Exec on success
    """

    title = 'Retry'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='max_retries', default=3),
        NodeInputType(label='initial_delay', default=1.0),
        NodeInputType(label='max_delay', default=30.0),
        NodeInputType(label='backoff_factor', default=2.0),
        NodeInputType(type_='exec', label='start'),
        NodeInputType(type_='exec', label='success'),
        NodeInputType(type_='exec', label='failure'),
    ]
    init_outputs = [
        NodeOutputType(label='attempt'),
        NodeOutputType(label='delay'),
        NodeOutputType(type_='exec', label='try'),
        NodeOutputType(type_='exec', label='exhausted'),
        NodeOutputType(type_='exec', label='succeeded'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self._attempt = 0
        self._current_delay = 1.0

    def update_event(self, inp=-1):
        max_retries = 3
        mr_input = self.input(0)
        if mr_input and mr_input.payload is not None:
            max_retries = int(mr_input.payload)

        initial_delay = 1.0
        id_input = self.input(1)
        if id_input and id_input.payload is not None:
            initial_delay = float(id_input.payload)

        max_delay = 30.0
        md_input = self.input(2)
        if md_input and md_input.payload is not None:
            max_delay = float(md_input.payload)

        backoff = 2.0
        bf_input = self.input(3)
        if bf_input and bf_input.payload is not None:
            backoff = float(bf_input.payload)

        if inp == 4:  # Start
            self._attempt = 0
            self._current_delay = initial_delay
            self.set_output_val(0, Data(self._attempt))
            self.set_output_val(1, Data(0))
            self.exec_output(2)  # try

        elif inp == 5:  # Success
            self.exec_output(4)  # succeeded

        elif inp == 6:  # Failure
            self._attempt += 1
            if self._attempt >= max_retries:
                self.set_output_val(0, Data(self._attempt))
                self.exec_output(3)  # exhausted
            else:
                self.set_output_val(0, Data(self._attempt))
                self.set_output_val(1, Data(self._current_delay))

                # Wait (in real impl, use async/timer)
                time.sleep(self._current_delay)

                self._current_delay = min(
                    self._current_delay * backoff,
                    max_delay
                )
                self.exec_output(2)  # retry

    def get_state(self) -> dict:
        return {
            'attempt': self._attempt,
            'current_delay': self._current_delay,
        }

    def set_state(self, data: dict, version):
        self._attempt = data.get('attempt', 0)
        self._current_delay = data.get('current_delay', 1.0)


class CircuitBreakerNode(Node):
    """
    Implement circuit breaker pattern.

    Prevents cascading failures by stopping requests
    to failing services.

    Inputs:
        - failure_threshold: Failures before opening circuit
        - reset_timeout: Seconds before trying again
        - exec_request: Trigger a request
        - exec_success: Signal success
        - exec_failure: Signal failure

    Outputs:
        - state: Current circuit state (closed, open, half-open)
        - failures: Current failure count
        - allowed: Exec when request allowed
        - blocked: Exec when request blocked
    """

    title = 'Circuit Breaker'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='failure_threshold', default=5),
        NodeInputType(label='reset_timeout', default=30.0),
        NodeInputType(type_='exec', label='request'),
        NodeInputType(type_='exec', label='success'),
        NodeInputType(type_='exec', label='failure'),
    ]
    init_outputs = [
        NodeOutputType(label='state'),
        NodeOutputType(label='failures'),
        NodeOutputType(type_='exec', label='allowed'),
        NodeOutputType(type_='exec', label='blocked'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self._state = 'closed'  # closed, open, half-open
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None

    def update_event(self, inp=-1):
        threshold = 5
        th_input = self.input(0)
        if th_input and th_input.payload is not None:
            threshold = int(th_input.payload)

        reset_timeout = 30.0
        rt_input = self.input(1)
        if rt_input and rt_input.payload is not None:
            reset_timeout = float(rt_input.payload)

        if inp == 2:  # Request
            now = time.time()

            # Check if we should transition from open to half-open
            if self._state == 'open' and self._last_failure_time:
                if now - self._last_failure_time >= reset_timeout:
                    self._state = 'half-open'

            self.set_output_val(0, Data(self._state))
            self.set_output_val(1, Data(self._failure_count))

            if self._state == 'open':
                self.exec_output(3)  # blocked
            else:
                self.exec_output(2)  # allowed

        elif inp == 3:  # Success
            self._failure_count = 0
            self._state = 'closed'
            self.set_output_val(0, Data(self._state))
            self.set_output_val(1, Data(self._failure_count))

        elif inp == 4:  # Failure
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._failure_count >= threshold:
                self._state = 'open'

            self.set_output_val(0, Data(self._state))
            self.set_output_val(1, Data(self._failure_count))

    def get_state(self) -> dict:
        return {
            'state': self._state,
            'failure_count': self._failure_count,
            'last_failure_time': self._last_failure_time,
        }

    def set_state(self, data: dict, version):
        self._state = data.get('state', 'closed')
        self._failure_count = data.get('failure_count', 0)
        self._last_failure_time = data.get('last_failure_time')


# Export load balancer nodes
load_balancer_nodes = [
    LoadBalancerNode,
    FallbackChainNode,
    RetryNode,
    CircuitBreakerNode,
]
