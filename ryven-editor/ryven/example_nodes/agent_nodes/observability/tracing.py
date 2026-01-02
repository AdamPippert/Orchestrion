"""
Tracing nodes for debugging and understanding agent behavior.
"""

from __future__ import annotations
import uuid
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from ryven.node_env import *


class TraceNode(Node):
    """
    Create and manage execution traces.

    Records the full execution path of a workflow
    for debugging and analysis.

    Inputs:
        - name: Trace name
        - metadata: Additional trace metadata
        - exec_start: Start a new trace
        - exec_end: End current trace

    Outputs:
        - trace_id: Current trace ID
        - trace: Complete trace data
        - done: Exec when trace ends
    """

    title = 'Trace'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='name'),
        NodeInputType(label='metadata'),
        NodeInputType(type_='exec', label='start'),
        NodeInputType(type_='exec', label='end'),
    ]
    init_outputs = [
        NodeOutputType(label='trace_id'),
        NodeOutputType(label='trace'),
        NodeOutputType(type_='exec', label='done'),
    ]

    # Class-level trace storage
    _traces: Dict[str, Dict[str, Any]] = {}
    _current_trace: Optional[str] = None

    def __init__(self, params):
        super().__init__(params)

    @classmethod
    def get_current_trace_id(cls) -> Optional[str]:
        return cls._current_trace

    @classmethod
    def get_trace(cls, trace_id: str) -> Optional[Dict[str, Any]]:
        return cls._traces.get(trace_id)

    @classmethod
    def add_span(cls, span: Dict[str, Any]) -> None:
        if cls._current_trace and cls._current_trace in cls._traces:
            cls._traces[cls._current_trace]['spans'].append(span)

    def update_event(self, inp=-1):
        if inp == 2:  # Start trace
            name = "trace"
            name_input = self.input(0)
            if name_input and name_input.payload:
                name = str(name_input.payload)

            metadata = {}
            meta_input = self.input(1)
            if meta_input and meta_input.payload:
                metadata = meta_input.payload

            trace_id = str(uuid.uuid4())
            TraceNode._current_trace = trace_id

            trace = {
                'id': trace_id,
                'name': name,
                'metadata': metadata,
                'started_at': datetime.now().isoformat(),
                'ended_at': None,
                'duration_ms': None,
                'spans': [],
                'status': 'running',
            }
            self._traces[trace_id] = trace

            self.set_output_val(0, Data(trace_id))
            self.set_output_val(1, Data(trace))

        elif inp == 3:  # End trace
            if TraceNode._current_trace:
                trace = self._traces.get(TraceNode._current_trace)
                if trace:
                    trace['ended_at'] = datetime.now().isoformat()
                    start = datetime.fromisoformat(trace['started_at'])
                    end = datetime.fromisoformat(trace['ended_at'])
                    trace['duration_ms'] = (end - start).total_seconds() * 1000
                    trace['status'] = 'completed'

                    self.set_output_val(0, Data(TraceNode._current_trace))
                    self.set_output_val(1, Data(trace))

                TraceNode._current_trace = None

            self.exec_output(2)


class SpanNode(Node):
    """
    Add a span to the current trace.

    Spans represent individual operations within a trace.

    Inputs:
        - name: Span name
        - type: Operation type (llm, tool, retrieval, etc.)
        - input_data: Input to the operation
        - output_data: Output from the operation
        - exec_start: Start span
        - exec_end: End span

    Outputs:
        - span_id: Span ID
        - duration_ms: Span duration
        - done: Exec
    """

    title = 'Span'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='name'),
        NodeInputType(label='type'),
        NodeInputType(label='input_data'),
        NodeInputType(label='output_data'),
        NodeInputType(type_='exec', label='start'),
        NodeInputType(type_='exec', label='end'),
    ]
    init_outputs = [
        NodeOutputType(label='span_id'),
        NodeOutputType(label='duration_ms'),
        NodeOutputType(type_='exec', label='done'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self._current_span: Optional[Dict[str, Any]] = None
        self._start_time: Optional[float] = None

    def update_event(self, inp=-1):
        if inp == 4:  # Start span
            name = "span"
            name_input = self.input(0)
            if name_input and name_input.payload:
                name = str(name_input.payload)

            span_type = "generic"
            type_input = self.input(1)
            if type_input and type_input.payload:
                span_type = str(type_input.payload)

            input_data = None
            input_input = self.input(2)
            if input_input:
                input_data = input_input.payload

            span_id = str(uuid.uuid4())
            self._start_time = time.time()

            self._current_span = {
                'id': span_id,
                'name': name,
                'type': span_type,
                'input': input_data,
                'output': None,
                'started_at': datetime.now().isoformat(),
                'ended_at': None,
                'duration_ms': None,
                'trace_id': TraceNode.get_current_trace_id(),
            }

            self.set_output_val(0, Data(span_id))

        elif inp == 5:  # End span
            if self._current_span:
                output_input = self.input(3)
                if output_input:
                    self._current_span['output'] = output_input.payload

                self._current_span['ended_at'] = datetime.now().isoformat()
                if self._start_time:
                    duration = (time.time() - self._start_time) * 1000
                    self._current_span['duration_ms'] = duration
                    self.set_output_val(1, Data(duration))

                # Add to parent trace
                TraceNode.add_span(self._current_span)

                self._current_span = None
                self._start_time = None

            self.exec_output(2)


class TraceViewerNode(Node):
    """
    View and analyze traces.

    Provides access to trace data for visualization
    and debugging.

    Inputs:
        - trace_id: Specific trace to view
        - filter_type: Filter spans by type
        - limit: Maximum traces to return
        - exec: Trigger

    Outputs:
        - trace: Requested trace data
        - all_traces: All stored traces
        - stats: Trace statistics
        - done: Exec
    """

    title = 'Trace Viewer'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='trace_id'),
        NodeInputType(label='filter_type'),
        NodeInputType(label='limit', default=10),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='trace'),
        NodeOutputType(label='all_traces'),
        NodeOutputType(label='stats'),
        NodeOutputType(type_='exec', label='done'),
    ]

    def update_event(self, inp=-1):
        if inp != 3:
            return

        trace_id = None
        id_input = self.input(0)
        if id_input and id_input.payload:
            trace_id = str(id_input.payload)

        filter_type = None
        filter_input = self.input(1)
        if filter_input and filter_input.payload:
            filter_type = str(filter_input.payload)

        limit = 10
        limit_input = self.input(2)
        if limit_input and limit_input.payload:
            limit = int(limit_input.payload)

        # Get specific trace or all
        if trace_id:
            trace = TraceNode.get_trace(trace_id)

            # Filter spans if requested
            if trace and filter_type:
                trace = dict(trace)
                trace['spans'] = [
                    s for s in trace['spans']
                    if s.get('type') == filter_type
                ]

            self.set_output_val(0, Data(trace))
        else:
            self.set_output_val(0, Data(None))

        # Get all traces
        all_traces = list(TraceNode._traces.values())[-limit:]
        self.set_output_val(1, Data(all_traces))

        # Calculate stats
        stats = {
            'total_traces': len(TraceNode._traces),
            'avg_duration_ms': 0,
            'span_type_counts': {},
        }

        durations = [t.get('duration_ms', 0) for t in all_traces if t.get('duration_ms')]
        if durations:
            stats['avg_duration_ms'] = sum(durations) / len(durations)

        for trace in all_traces:
            for span in trace.get('spans', []):
                span_type = span.get('type', 'unknown')
                stats['span_type_counts'][span_type] = \
                    stats['span_type_counts'].get(span_type, 0) + 1

        self.set_output_val(2, Data(stats))
        self.exec_output(3)


# Export tracing nodes
tracing_nodes = [
    TraceNode,
    SpanNode,
    TraceViewerNode,
]
