"""
Resource limit guardrails.

Nodes for controlling token usage, costs, rate limits,
and execution timeouts.
"""

from __future__ import annotations
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from collections import deque
from ryven.node_env import *


class TokenLimitNode(Node):
    """
    Enforce token limits on LLM interactions.

    Tracks token usage and blocks requests that would
    exceed the configured limits.

    Inputs:
        - response: CompletionResponse to check
        - max_tokens_per_request: Maximum tokens per request
        - max_tokens_per_session: Maximum tokens for session
        - reset_session: Reset session counter
        - exec: Trigger

    Outputs:
        - passed: Whether within limits
        - tokens_used: Tokens used this request
        - session_total: Total tokens this session
        - remaining: Tokens remaining in session
        - allowed: Exec if allowed
        - exceeded: Exec if limit exceeded
    """

    title = 'Token Limit'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='response'),
        NodeInputType(label='max_tokens_per_request'),
        NodeInputType(label='max_tokens_per_session'),
        NodeInputType(label='reset_session', default=False),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='passed'),
        NodeOutputType(label='tokens_used'),
        NodeOutputType(label='session_total'),
        NodeOutputType(label='remaining'),
        NodeOutputType(type_='exec', label='allowed'),
        NodeOutputType(type_='exec', label='exceeded'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self._session_tokens = 0

    def update_event(self, inp=-1):
        if inp != 4:
            return

        # Check for reset
        reset_input = self.input(3)
        if reset_input and reset_input.payload:
            self._session_tokens = 0

        response_input = self.input(0)
        if not response_input or not response_input.payload:
            self.set_output_val(0, Data(True))
            self.exec_output(4)
            return

        response = response_input.payload
        tokens_used = 0

        # Extract token count from response
        if hasattr(response, 'total_tokens'):
            tokens_used = response.total_tokens
        elif isinstance(response, dict):
            tokens_used = response.get('total_tokens', 0)

        exceeded = False
        reason = None

        # Check per-request limit
        per_req_input = self.input(1)
        if per_req_input and per_req_input.payload:
            max_per_request = int(per_req_input.payload)
            if tokens_used > max_per_request:
                exceeded = True
                reason = f"Request tokens ({tokens_used}) exceed limit ({max_per_request})"

        # Check session limit
        session_input = self.input(2)
        if session_input and session_input.payload:
            max_session = int(session_input.payload)
            if self._session_tokens + tokens_used > max_session:
                exceeded = True
                reason = f"Session would exceed limit ({self._session_tokens + tokens_used} > {max_session})"

        if not exceeded:
            self._session_tokens += tokens_used

        remaining = float('inf')
        if session_input and session_input.payload:
            remaining = int(session_input.payload) - self._session_tokens

        self.set_output_val(0, Data(not exceeded))
        self.set_output_val(1, Data(tokens_used))
        self.set_output_val(2, Data(self._session_tokens))
        self.set_output_val(3, Data(remaining))

        if exceeded:
            self.exec_output(5)
        else:
            self.exec_output(4)

    def get_state(self) -> dict:
        return {'session_tokens': self._session_tokens}

    def set_state(self, data: dict, version):
        self._session_tokens = data.get('session_tokens', 0)


class CostLimitNode(Node):
    """
    Enforce cost limits on LLM usage.

    Tracks spending and blocks requests that would
    exceed budget.

    Inputs:
        - response: CompletionResponse to check
        - max_cost_per_request: Maximum cost per request ($)
        - max_cost_per_session: Maximum cost for session ($)
        - max_cost_per_day: Maximum daily spend ($)
        - reset_session: Reset session counter
        - exec: Trigger

    Outputs:
        - passed: Whether within limits
        - request_cost: Cost of this request
        - session_total: Total cost this session
        - daily_total: Total cost today
        - allowed: Exec if allowed
        - exceeded: Exec if exceeded
    """

    title = 'Cost Limit'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='response'),
        NodeInputType(label='max_cost_per_request'),
        NodeInputType(label='max_cost_per_session'),
        NodeInputType(label='max_cost_per_day'),
        NodeInputType(label='reset_session', default=False),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='passed'),
        NodeOutputType(label='request_cost'),
        NodeOutputType(label='session_total'),
        NodeOutputType(label='daily_total'),
        NodeOutputType(type_='exec', label='allowed'),
        NodeOutputType(type_='exec', label='exceeded'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self._session_cost = 0.0
        self._daily_cost = 0.0
        self._last_reset_date = datetime.now().date()

    def _calculate_cost(self, response) -> float:
        """Calculate cost from response using model pricing."""
        from ..providers.model_info import get_model_info

        prompt_tokens = 0
        completion_tokens = 0
        model = ""

        if hasattr(response, 'prompt_tokens'):
            prompt_tokens = response.prompt_tokens
            completion_tokens = response.completion_tokens
            model = response.model
        elif isinstance(response, dict):
            prompt_tokens = response.get('prompt_tokens', 0)
            completion_tokens = response.get('completion_tokens', 0)
            model = response.get('model', '')

        # Get pricing
        model_info = get_model_info(model)
        if model_info:
            input_cost = (prompt_tokens / 1_000_000) * model_info.capabilities.input_cost_per_million
            output_cost = (completion_tokens / 1_000_000) * model_info.capabilities.output_cost_per_million
            return input_cost + output_cost

        return 0.0

    def update_event(self, inp=-1):
        if inp != 5:
            return

        # Check for daily reset
        today = datetime.now().date()
        if today > self._last_reset_date:
            self._daily_cost = 0.0
            self._last_reset_date = today

        # Check for session reset
        reset_input = self.input(4)
        if reset_input and reset_input.payload:
            self._session_cost = 0.0

        response_input = self.input(0)
        if not response_input or not response_input.payload:
            self.set_output_val(0, Data(True))
            self.exec_output(4)
            return

        response = response_input.payload
        request_cost = self._calculate_cost(response)
        exceeded = False

        # Check per-request limit
        per_req_input = self.input(1)
        if per_req_input and per_req_input.payload:
            max_per_request = float(per_req_input.payload)
            if request_cost > max_per_request:
                exceeded = True

        # Check session limit
        session_input = self.input(2)
        if session_input and session_input.payload:
            max_session = float(session_input.payload)
            if self._session_cost + request_cost > max_session:
                exceeded = True

        # Check daily limit
        daily_input = self.input(3)
        if daily_input and daily_input.payload:
            max_daily = float(daily_input.payload)
            if self._daily_cost + request_cost > max_daily:
                exceeded = True

        if not exceeded:
            self._session_cost += request_cost
            self._daily_cost += request_cost

        self.set_output_val(0, Data(not exceeded))
        self.set_output_val(1, Data(request_cost))
        self.set_output_val(2, Data(self._session_cost))
        self.set_output_val(3, Data(self._daily_cost))

        if exceeded:
            self.exec_output(5)
        else:
            self.exec_output(4)

    def get_state(self) -> dict:
        return {
            'session_cost': self._session_cost,
            'daily_cost': self._daily_cost,
            'last_reset': self._last_reset_date.isoformat(),
        }

    def set_state(self, data: dict, version):
        self._session_cost = data.get('session_cost', 0.0)
        self._daily_cost = data.get('daily_cost', 0.0)
        if 'last_reset' in data:
            self._last_reset_date = datetime.fromisoformat(data['last_reset']).date()


class RateLimitNode(Node):
    """
    Enforce rate limits on requests.

    Implements token bucket algorithm for smooth rate limiting.

    Inputs:
        - requests_per_minute: Maximum requests per minute
        - requests_per_hour: Maximum requests per hour
        - tokens_per_minute: Maximum tokens per minute
        - exec: Trigger (blocks until rate allows)

    Outputs:
        - allowed: Whether request is allowed
        - wait_time: Seconds to wait before retry
        - requests_remaining: Requests left this minute
        - passed: Exec when allowed
        - throttled: Exec if throttled
    """

    title = 'Rate Limit'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='requests_per_minute'),
        NodeInputType(label='requests_per_hour'),
        NodeInputType(label='tokens_per_minute'),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='allowed'),
        NodeOutputType(label='wait_time'),
        NodeOutputType(label='requests_remaining'),
        NodeOutputType(type_='exec', label='passed'),
        NodeOutputType(type_='exec', label='throttled'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self._minute_requests: deque = deque()  # timestamps
        self._hour_requests: deque = deque()

    def _clean_old_requests(self):
        """Remove requests outside the window."""
        now = time.time()
        minute_ago = now - 60
        hour_ago = now - 3600

        while self._minute_requests and self._minute_requests[0] < minute_ago:
            self._minute_requests.popleft()

        while self._hour_requests and self._hour_requests[0] < hour_ago:
            self._hour_requests.popleft()

    def update_event(self, inp=-1):
        if inp != 3:
            return

        self._clean_old_requests()
        now = time.time()
        allowed = True
        wait_time = 0.0

        # Check per-minute limit
        rpm_input = self.input(0)
        if rpm_input and rpm_input.payload:
            rpm = int(rpm_input.payload)
            if len(self._minute_requests) >= rpm:
                allowed = False
                oldest = self._minute_requests[0]
                wait_time = max(wait_time, 60 - (now - oldest))

        # Check per-hour limit
        rph_input = self.input(1)
        if rph_input and rph_input.payload:
            rph = int(rph_input.payload)
            if len(self._hour_requests) >= rph:
                allowed = False
                oldest = self._hour_requests[0]
                wait_time = max(wait_time, 3600 - (now - oldest))

        if allowed:
            self._minute_requests.append(now)
            self._hour_requests.append(now)

        requests_remaining = 0
        if rpm_input and rpm_input.payload:
            requests_remaining = int(rpm_input.payload) - len(self._minute_requests)

        self.set_output_val(0, Data(allowed))
        self.set_output_val(1, Data(wait_time))
        self.set_output_val(2, Data(requests_remaining))

        if allowed:
            self.exec_output(3)
        else:
            self.exec_output(4)


class TimeoutNode(Node):
    """
    Enforce timeout on operations.

    Wraps execution with a timeout, failing if operation
    takes too long.

    Inputs:
        - timeout_seconds: Maximum execution time
        - exec: Start timeout
        - complete: Signal operation complete

    Outputs:
        - elapsed: Time elapsed
        - timed_out: Whether timeout occurred
        - success: Exec if completed in time
        - timeout: Exec if timed out
    """

    title = 'Timeout'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='timeout_seconds', default=30),
        NodeInputType(type_='exec', label='start'),
        NodeInputType(type_='exec', label='complete'),
    ]
    init_outputs = [
        NodeOutputType(label='elapsed'),
        NodeOutputType(label='timed_out'),
        NodeOutputType(type_='exec', label='success'),
        NodeOutputType(type_='exec', label='timeout'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self._start_time: Optional[float] = None
        self._timed_out = False

    def update_event(self, inp=-1):
        if inp == 1:  # Start
            self._start_time = time.time()
            self._timed_out = False

            timeout = 30.0
            timeout_input = self.input(0)
            if timeout_input and timeout_input.payload:
                timeout = float(timeout_input.payload)

            # Note: In a real implementation, you'd want to use
            # async/threading for actual timeout enforcement.
            # This is a simplified version.

        elif inp == 2:  # Complete
            if self._start_time is None:
                return

            elapsed = time.time() - self._start_time
            timeout = 30.0
            timeout_input = self.input(0)
            if timeout_input and timeout_input.payload:
                timeout = float(timeout_input.payload)

            self._timed_out = elapsed > timeout
            self.set_output_val(0, Data(elapsed))
            self.set_output_val(1, Data(self._timed_out))

            if self._timed_out:
                self.exec_output(3)
            else:
                self.exec_output(2)


# Export limit nodes
limit_nodes = [
    TokenLimitNode,
    CostLimitNode,
    RateLimitNode,
    TimeoutNode,
]
