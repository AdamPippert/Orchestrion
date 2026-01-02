"""
Orchestration pattern nodes.

Implement common workflow patterns for agent coordination.
"""

from __future__ import annotations
import asyncio
from typing import List, Dict, Any, Optional, Callable
from ryven.node_env import *


class ParallelNode(Node):
    """
    Execute multiple operations in parallel.

    Fans out to multiple paths and waits for all to complete.

    Inputs:
        - inputs (dynamic): Multiple input values to process
        - exec: Trigger parallel execution

    Outputs:
        - results: List of all results
        - first: First result to complete
        - errors: Any errors that occurred
        - all_done: Exec when all complete
    """

    title = 'Parallel'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='results'),
        NodeOutputType(label='first'),
        NodeOutputType(label='errors'),
        NodeOutputType(type_='exec', label='all_done'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self.num_data_inputs = 0

    def place_event(self):
        if self.num_data_inputs == 0:
            self.add_data_input()
            self.add_data_input()

    def add_data_input(self):
        self.create_input(label=f'input_{self.num_data_inputs}')
        self.num_data_inputs += 1

    def remove_data_input(self, index):
        self.delete_input(index)
        self.num_data_inputs -= 1

    def update_event(self, inp=-1):
        if inp != 0:  # Only trigger on exec
            return

        results = []
        errors = []

        # Collect all input values
        for i in range(1, len(self.inputs)):
            val_input = self.input(i)
            if val_input and val_input.payload is not None:
                results.append(val_input.payload)

        self.set_output_val(0, Data(results))
        self.set_output_val(1, Data(results[0] if results else None))
        self.set_output_val(2, Data(errors))
        self.exec_output(3)

    def get_state(self) -> dict:
        return {'num_data_inputs': self.num_data_inputs}

    def set_state(self, data: dict, version):
        self.num_data_inputs = data.get('num_data_inputs', 0)


class SequenceNode(Node):
    """
    Execute operations in sequence.

    Chains operations where each depends on the previous.

    Inputs:
        - initial: Initial value to pass through
        - exec: Start the sequence

    Outputs:
        - value: Current value in sequence
        - step: Current step number
        - step_0...step_n: Exec outputs for each step
        - done: Exec when sequence complete
    """

    title = 'Sequence'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='initial'),
        NodeInputType(label='num_steps', default=3),
        NodeInputType(type_='exec', label='start'),
        NodeInputType(type_='exec', label='continue'),
    ]
    init_outputs = [
        NodeOutputType(label='value'),
        NodeOutputType(label='step'),
        NodeOutputType(type_='exec', label='step_0'),
        NodeOutputType(type_='exec', label='step_1'),
        NodeOutputType(type_='exec', label='step_2'),
        NodeOutputType(type_='exec', label='done'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self._current_step = 0
        self._current_value = None

    def update_event(self, inp=-1):
        num_steps = 3
        ns_input = self.input(1)
        if ns_input and ns_input.payload is not None:
            num_steps = int(ns_input.payload)

        if inp == 2:  # Start
            initial = None
            init_input = self.input(0)
            if init_input:
                initial = init_input.payload

            self._current_step = 0
            self._current_value = initial

            self.set_output_val(0, Data(self._current_value))
            self.set_output_val(1, Data(self._current_step))

            if self._current_step < num_steps:
                self.exec_output(2 + self._current_step)  # step_N

        elif inp == 3:  # Continue
            self._current_step += 1

            # Get updated value from flow
            init_input = self.input(0)
            if init_input:
                self._current_value = init_input.payload

            self.set_output_val(0, Data(self._current_value))
            self.set_output_val(1, Data(self._current_step))

            if self._current_step < num_steps:
                step_output = 2 + min(self._current_step, 2)  # Cap at step_2
                self.exec_output(step_output)
            else:
                self.exec_output(5)  # done

    def get_state(self) -> dict:
        return {
            'current_step': self._current_step,
            'current_value': self._current_value,
        }

    def set_state(self, data: dict, version):
        self._current_step = data.get('current_step', 0)
        self._current_value = data.get('current_value')


class ConditionalNode(Node):
    """
    Branch execution based on condition.

    Routes to different paths based on a boolean condition.

    Inputs:
        - condition: Boolean or value to evaluate
        - value: Value to pass through
        - exec: Trigger evaluation

    Outputs:
        - value: The input value (passed through)
        - is_true: Boolean result of condition
        - then: Exec if condition is true
        - else: Exec if condition is false
    """

    title = 'Conditional'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='condition'),
        NodeInputType(label='value'),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='value'),
        NodeOutputType(label='is_true'),
        NodeOutputType(type_='exec', label='then'),
        NodeOutputType(type_='exec', label='else'),
    ]

    def update_event(self, inp=-1):
        if inp != 2:
            return

        condition = False
        cond_input = self.input(0)
        if cond_input and cond_input.payload is not None:
            condition = bool(cond_input.payload)

        value = None
        val_input = self.input(1)
        if val_input:
            value = val_input.payload

        self.set_output_val(0, Data(value))
        self.set_output_val(1, Data(condition))

        if condition:
            self.exec_output(2)  # then
        else:
            self.exec_output(3)  # else


class LoopNode(Node):
    """
    Iterate over a collection or count.

    Executes body for each item in collection or N times.

    Inputs:
        - items: Collection to iterate over (or count)
        - exec: Start loop

    Outputs:
        - item: Current item
        - index: Current index
        - body: Exec for each iteration
        - done: Exec when loop complete
    """

    title = 'Loop'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='items'),
        NodeInputType(type_='exec', label='start'),
        NodeInputType(type_='exec', label='next'),
    ]
    init_outputs = [
        NodeOutputType(label='item'),
        NodeOutputType(label='index'),
        NodeOutputType(label='total'),
        NodeOutputType(type_='exec', label='body'),
        NodeOutputType(type_='exec', label='done'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self._items: List[Any] = []
        self._index = 0

    def update_event(self, inp=-1):
        if inp == 1:  # Start
            items_input = self.input(0)
            if items_input and items_input.payload is not None:
                items = items_input.payload
                if isinstance(items, int):
                    self._items = list(range(items))
                elif isinstance(items, (list, tuple)):
                    self._items = list(items)
                else:
                    self._items = [items]
            else:
                self._items = []

            self._index = 0

            if self._items:
                self.set_output_val(0, Data(self._items[0]))
                self.set_output_val(1, Data(0))
                self.set_output_val(2, Data(len(self._items)))
                self.exec_output(3)  # body
            else:
                self.set_output_val(2, Data(0))
                self.exec_output(4)  # done

        elif inp == 2:  # Next
            self._index += 1

            if self._index < len(self._items):
                self.set_output_val(0, Data(self._items[self._index]))
                self.set_output_val(1, Data(self._index))
                self.exec_output(3)  # body
            else:
                self.exec_output(4)  # done

    def get_state(self) -> dict:
        return {
            'items': self._items,
            'index': self._index,
        }

    def set_state(self, data: dict, version):
        self._items = data.get('items', [])
        self._index = data.get('index', 0)


class MapNode(Node):
    """
    Apply an operation to each item in a collection.

    Like Loop but collects results automatically.

    Inputs:
        - items: Collection to map over
        - result: Result from current iteration
        - exec_start: Start mapping
        - exec_next: Signal iteration complete

    Outputs:
        - item: Current item being processed
        - index: Current index
        - results: All collected results
        - process: Exec to process each item
        - done: Exec when all items processed
    """

    title = 'Map'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='items'),
        NodeInputType(label='result'),
        NodeInputType(type_='exec', label='start'),
        NodeInputType(type_='exec', label='next'),
    ]
    init_outputs = [
        NodeOutputType(label='item'),
        NodeOutputType(label='index'),
        NodeOutputType(label='results'),
        NodeOutputType(type_='exec', label='process'),
        NodeOutputType(type_='exec', label='done'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self._items: List[Any] = []
        self._results: List[Any] = []
        self._index = 0

    def update_event(self, inp=-1):
        if inp == 2:  # Start
            items_input = self.input(0)
            if items_input and items_input.payload is not None:
                items = items_input.payload
                if isinstance(items, (list, tuple)):
                    self._items = list(items)
                else:
                    self._items = [items]
            else:
                self._items = []

            self._results = []
            self._index = 0

            if self._items:
                self.set_output_val(0, Data(self._items[0]))
                self.set_output_val(1, Data(0))
                self.exec_output(3)  # process
            else:
                self.set_output_val(2, Data([]))
                self.exec_output(4)  # done

        elif inp == 3:  # Next
            # Collect result from this iteration
            result_input = self.input(1)
            if result_input:
                self._results.append(result_input.payload)

            self._index += 1

            if self._index < len(self._items):
                self.set_output_val(0, Data(self._items[self._index]))
                self.set_output_val(1, Data(self._index))
                self.exec_output(3)  # process
            else:
                self.set_output_val(2, Data(self._results))
                self.exec_output(4)  # done

    def get_state(self) -> dict:
        return {
            'items': self._items,
            'results': self._results,
            'index': self._index,
        }

    def set_state(self, data: dict, version):
        self._items = data.get('items', [])
        self._results = data.get('results', [])
        self._index = data.get('index', 0)


class ReduceNode(Node):
    """
    Reduce a collection to a single value.

    Accumulates results across iterations.

    Inputs:
        - items: Collection to reduce
        - accumulator: Current accumulated value
        - initial: Initial accumulator value
        - exec_start: Start reduction
        - exec_next: Signal iteration complete

    Outputs:
        - item: Current item
        - acc: Current accumulator value
        - result: Final reduced value
        - process: Exec for each item
        - done: Exec when complete
    """

    title = 'Reduce'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='items'),
        NodeInputType(label='accumulator'),
        NodeInputType(label='initial'),
        NodeInputType(type_='exec', label='start'),
        NodeInputType(type_='exec', label='next'),
    ]
    init_outputs = [
        NodeOutputType(label='item'),
        NodeOutputType(label='acc'),
        NodeOutputType(label='result'),
        NodeOutputType(type_='exec', label='process'),
        NodeOutputType(type_='exec', label='done'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self._items: List[Any] = []
        self._acc: Any = None
        self._index = 0

    def update_event(self, inp=-1):
        if inp == 3:  # Start
            items_input = self.input(0)
            if items_input and items_input.payload is not None:
                self._items = list(items_input.payload) if isinstance(items_input.payload, (list, tuple)) else [items_input.payload]
            else:
                self._items = []

            initial_input = self.input(2)
            self._acc = initial_input.payload if initial_input else None
            self._index = 0

            if self._items:
                self.set_output_val(0, Data(self._items[0]))
                self.set_output_val(1, Data(self._acc))
                self.exec_output(3)  # process
            else:
                self.set_output_val(2, Data(self._acc))
                self.exec_output(4)  # done

        elif inp == 4:  # Next
            acc_input = self.input(1)
            if acc_input:
                self._acc = acc_input.payload

            self._index += 1

            if self._index < len(self._items):
                self.set_output_val(0, Data(self._items[self._index]))
                self.set_output_val(1, Data(self._acc))
                self.exec_output(3)  # process
            else:
                self.set_output_val(2, Data(self._acc))
                self.exec_output(4)  # done


# Export orchestration nodes
orchestration_nodes = [
    ParallelNode,
    SequenceNode,
    ConditionalNode,
    LoopNode,
    MapNode,
    ReduceNode,
]
