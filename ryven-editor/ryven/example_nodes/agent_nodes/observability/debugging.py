"""
Debugging and inspection nodes.
"""

from __future__ import annotations
import json
from typing import Any, Dict, Optional
from datetime import datetime
from ryven.node_env import *


class DebugNode(Node):
    """
    Debug output node.

    Logs data passing through for debugging purposes.

    Inputs:
        - data: Data to debug
        - label: Label for the debug output
        - verbose: Show full data or summary
        - exec: Trigger

    Outputs:
        - data: Pass-through of input data
        - debug_output: Formatted debug string
        - done: Exec
    """

    title = 'Debug'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='data'),
        NodeInputType(label='label'),
        NodeInputType(label='verbose', default=False),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='data'),
        NodeOutputType(label='debug_output'),
        NodeOutputType(type_='exec', label='done'),
    ]

    # Class-level debug log
    _debug_log: list = []

    @classmethod
    def get_log(cls) -> list:
        return cls._debug_log

    @classmethod
    def clear_log(cls) -> None:
        cls._debug_log = []

    def _format_data(self, data: Any, verbose: bool) -> str:
        """Format data for debug output."""
        if data is None:
            return "None"

        if isinstance(data, str):
            if verbose or len(data) < 200:
                return f'"{data}"'
            return f'"{data[:100]}..." ({len(data)} chars)'

        if isinstance(data, (int, float, bool)):
            return str(data)

        if isinstance(data, list):
            if verbose:
                try:
                    return json.dumps(data, indent=2, default=str)
                except Exception:
                    return str(data)
            return f"[List of {len(data)} items]"

        if isinstance(data, dict):
            if verbose:
                try:
                    return json.dumps(data, indent=2, default=str)
                except Exception:
                    return str(data)
            return f"{{Dict with keys: {list(data.keys())[:5]}...}}"

        # For objects, show type and basic info
        type_name = type(data).__name__
        if hasattr(data, '__dict__'):
            attrs = list(data.__dict__.keys())[:5]
            return f"<{type_name} with attrs: {attrs}>"
        return f"<{type_name}>"

    def update_event(self, inp=-1):
        if inp != 3:
            return

        data = None
        data_input = self.input(0)
        if data_input:
            data = data_input.payload

        label = "debug"
        label_input = self.input(1)
        if label_input and label_input.payload:
            label = str(label_input.payload)

        verbose = False
        verbose_input = self.input(2)
        if verbose_input and verbose_input.payload is not None:
            verbose = bool(verbose_input.payload)

        formatted = self._format_data(data, verbose)
        debug_output = f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {label}: {formatted}"

        # Add to log
        self._debug_log.append({
            'timestamp': datetime.now().isoformat(),
            'label': label,
            'data': data,
            'formatted': formatted,
        })

        # Also print to console
        print(debug_output)

        self.set_output_val(0, Data(data))
        self.set_output_val(1, Data(debug_output))
        self.exec_output(2)


class InspectorNode(Node):
    """
    Inspect data structure and type.

    Provides detailed information about data
    flowing through the workflow.

    Inputs:
        - data: Data to inspect
        - exec: Trigger

    Outputs:
        - type: Type of the data
        - structure: Structure information
        - size: Size/length information
        - schema: Inferred schema (for dicts/objects)
        - done: Exec
    """

    title = 'Inspector'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='data'),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='type'),
        NodeOutputType(label='structure'),
        NodeOutputType(label='size'),
        NodeOutputType(label='schema'),
        NodeOutputType(type_='exec', label='done'),
    ]

    def _get_size(self, data: Any) -> Dict[str, Any]:
        """Get size information for data."""
        size_info = {'type': type(data).__name__}

        if isinstance(data, str):
            size_info['chars'] = len(data)
            size_info['estimated_tokens'] = len(data) // 4
        elif isinstance(data, (list, tuple)):
            size_info['items'] = len(data)
            if data:
                size_info['first_item_type'] = type(data[0]).__name__
        elif isinstance(data, dict):
            size_info['keys'] = len(data)
            size_info['key_names'] = list(data.keys())[:10]
        elif hasattr(data, '__len__'):
            size_info['length'] = len(data)

        return size_info

    def _infer_schema(self, data: Any, depth: int = 0) -> Dict[str, Any]:
        """Infer schema from data structure."""
        if depth > 3:  # Limit recursion
            return {'type': type(data).__name__}

        if data is None:
            return {'type': 'null'}
        elif isinstance(data, bool):
            return {'type': 'boolean', 'value': data}
        elif isinstance(data, int):
            return {'type': 'integer'}
        elif isinstance(data, float):
            return {'type': 'number'}
        elif isinstance(data, str):
            return {'type': 'string', 'length': len(data)}
        elif isinstance(data, list):
            if not data:
                return {'type': 'array', 'items': 'unknown'}
            return {
                'type': 'array',
                'items': self._infer_schema(data[0], depth + 1),
                'length': len(data),
            }
        elif isinstance(data, dict):
            properties = {}
            for key, value in list(data.items())[:10]:
                properties[key] = self._infer_schema(value, depth + 1)
            return {
                'type': 'object',
                'properties': properties,
            }
        else:
            schema = {'type': type(data).__name__}
            if hasattr(data, '__dict__'):
                schema['attributes'] = list(data.__dict__.keys())[:10]
            return schema

    def update_event(self, inp=-1):
        if inp != 1:
            return

        data = None
        data_input = self.input(0)
        if data_input:
            data = data_input.payload

        type_name = type(data).__name__ if data is not None else 'NoneType'
        structure = self._get_size(data)
        size = structure.get('chars') or structure.get('items') or structure.get('keys') or 0
        schema = self._infer_schema(data)

        self.set_output_val(0, Data(type_name))
        self.set_output_val(1, Data(structure))
        self.set_output_val(2, Data(size))
        self.set_output_val(3, Data(schema))
        self.exec_output(4)


class BreakpointNode(Node):
    """
    Conditional breakpoint for debugging.

    Pauses execution when condition is met,
    allowing inspection of data.

    Inputs:
        - data: Data to check
        - condition: Condition to evaluate (Python expression)
        - enabled: Whether breakpoint is active
        - exec: Trigger

    Outputs:
        - data: Pass-through data
        - triggered: Whether breakpoint was hit
        - continue: Exec to continue (after pause)
        - skip: Exec if condition not met
    """

    title = 'Breakpoint'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='data'),
        NodeInputType(label='condition'),
        NodeInputType(label='enabled', default=True),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='data'),
        NodeOutputType(label='triggered'),
        NodeOutputType(type_='exec', label='hit'),
        NodeOutputType(type_='exec', label='skip'),
    ]

    def update_event(self, inp=-1):
        if inp != 3:
            return

        data = None
        data_input = self.input(0)
        if data_input:
            data = data_input.payload

        enabled = True
        enabled_input = self.input(2)
        if enabled_input and enabled_input.payload is not None:
            enabled = bool(enabled_input.payload)

        if not enabled:
            self.set_output_val(0, Data(data))
            self.set_output_val(1, Data(False))
            self.exec_output(3)  # skip
            return

        # Evaluate condition
        condition_met = True
        condition_input = self.input(1)
        if condition_input and condition_input.payload:
            condition = str(condition_input.payload)
            try:
                # Simple condition evaluation
                # In production, use a safer expression evaluator
                condition_met = eval(condition, {'data': data})
            except Exception:
                condition_met = True  # Default to triggering

        self.set_output_val(0, Data(data))
        self.set_output_val(1, Data(condition_met))

        if condition_met:
            print(f"[BREAKPOINT] Triggered at {datetime.now().isoformat()}")
            print(f"[BREAKPOINT] Data: {data}")
            self.exec_output(2)  # hit
        else:
            self.exec_output(3)  # skip


# Export debugging nodes
debugging_nodes = [
    DebugNode,
    InspectorNode,
    BreakpointNode,
]
