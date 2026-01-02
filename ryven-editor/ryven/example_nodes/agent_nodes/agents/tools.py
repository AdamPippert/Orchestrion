"""
Tool definition and execution nodes.

Tools are functions that agents can call to interact with
the outside world or perform specific tasks.
"""

from __future__ import annotations
import json
from typing import Dict, Any, List, Optional, Callable
from ryven.node_env import *


class ToolDefinitionNode(Node):
    """
    Define a tool that an agent can use.

    Tools follow the OpenAI function calling format and can be
    used with any provider that supports function calling.

    Inputs:
        - name: Tool name (must be a valid identifier)
        - description: What the tool does (helps LLM decide when to use it)
        - parameters: JSON Schema defining the parameters
        - required: List of required parameter names
        - handler: Function to execute when tool is called

    Outputs:
        - tool: Tool definition in OpenAI format
        - schema: The full JSON schema
    """

    title = 'Tool Definition'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='name'),
        NodeInputType(label='description'),
        NodeInputType(label='parameters'),
        NodeInputType(label='required'),
        NodeInputType(label='handler'),
    ]
    init_outputs = [
        NodeOutputType(label='tool'),
        NodeOutputType(label='schema'),
    ]

    def update_event(self, inp=-1):
        name_input = self.input(0)
        if not name_input or not name_input.payload:
            return

        name = str(name_input.payload)

        description = ""
        desc_input = self.input(1)
        if desc_input and desc_input.payload:
            description = str(desc_input.payload)

        parameters = {"type": "object", "properties": {}}
        params_input = self.input(2)
        if params_input and params_input.payload:
            if isinstance(params_input.payload, dict):
                parameters = params_input.payload
            elif isinstance(params_input.payload, str):
                try:
                    parameters = json.loads(params_input.payload)
                except json.JSONDecodeError:
                    pass

        required = []
        req_input = self.input(3)
        if req_input and req_input.payload:
            if isinstance(req_input.payload, list):
                required = req_input.payload
            else:
                required = [str(req_input.payload)]

        handler = None
        handler_input = self.input(4)
        if handler_input and handler_input.payload:
            handler = handler_input.payload

        # Add required to parameters schema
        if required and "required" not in parameters:
            parameters["required"] = required

        # Build OpenAI-format tool definition
        tool = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
            "_handler": handler,  # Internal: the actual function to call
        }

        self.set_output_val(0, Data(tool))
        self.set_output_val(1, Data(parameters))


class ToolRegistryNode(Node):
    """
    Collect multiple tools into a registry for an agent.

    Provides a single output that can be connected to agent
    or chat nodes.

    Inputs:
        - tools (dynamic): Connect multiple tool definitions

    Outputs:
        - tools: List of all tools
        - tool_names: List of tool names
    """

    title = 'Tool Registry'
    version = 'v0.1'
    init_inputs = []
    init_outputs = [
        NodeOutputType(label='tools'),
        NodeOutputType(label='tool_names'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self.num_inputs = 0

    def place_event(self):
        if self.num_inputs == 0:
            self.add_tool_input()
            self.add_tool_input()

    def add_tool_input(self):
        self.create_input(label=f'tool_{self.num_inputs}')
        self.num_inputs += 1

    def remove_tool_input(self, index):
        self.delete_input(index)
        self.num_inputs -= 1

    def update_event(self, inp=-1):
        tools = []
        tool_names = []

        for i in range(len(self.inputs)):
            tool_input = self.input(i)
            if tool_input and tool_input.payload:
                tool = tool_input.payload
                if isinstance(tool, dict):
                    tools.append(tool)
                    if "function" in tool:
                        tool_names.append(tool["function"]["name"])
                elif isinstance(tool, list):
                    for t in tool:
                        if isinstance(t, dict):
                            tools.append(t)
                            if "function" in t:
                                tool_names.append(t["function"]["name"])

        self.set_output_val(0, Data(tools))
        self.set_output_val(1, Data(tool_names))

    def get_state(self) -> dict:
        return {'num_inputs': self.num_inputs}

    def set_state(self, data: dict, version):
        self.num_inputs = data.get('num_inputs', 0)


class ToolExecutorNode(Node):
    """
    Execute a tool call from an LLM response.

    Takes tool call information from a chat response and
    executes the corresponding tool, returning the result.

    Inputs:
        - tool_call: Tool call from LLM (dict with id, function, arguments)
        - tools: Tool registry with handlers
        - exec: Execution trigger

    Outputs:
        - result: Tool execution result
        - tool_message: Message to send back to LLM
        - error: Any error that occurred
        - done: Exec signal when complete
    """

    title = 'Tool Executor'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='tool_call'),
        NodeInputType(label='tools'),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='result'),
        NodeOutputType(label='tool_message'),
        NodeOutputType(label='error'),
        NodeOutputType(type_='exec', label='done'),
    ]

    def update_event(self, inp=-1):
        if inp != 2:  # Only trigger on exec
            return

        call_input = self.input(0)
        tools_input = self.input(1)

        if not call_input or not call_input.payload:
            self.set_output_val(2, Data("Error: No tool call provided"))
            self.exec_output(3)
            return

        if not tools_input or not tools_input.payload:
            self.set_output_val(2, Data("Error: No tools provided"))
            self.exec_output(3)
            return

        tool_call = call_input.payload
        tools = tools_input.payload

        # Find the matching tool
        func_name = tool_call.get("function", {}).get("name", "")
        arguments_str = tool_call.get("function", {}).get("arguments", "{}")
        tool_call_id = tool_call.get("id", "")

        handler = None
        for tool in tools:
            if tool.get("function", {}).get("name") == func_name:
                handler = tool.get("_handler")
                break

        if not handler:
            error = f"Error: No handler found for tool '{func_name}'"
            self.set_output_val(2, Data(error))
            self.exec_output(3)
            return

        # Parse arguments
        try:
            if isinstance(arguments_str, str):
                arguments = json.loads(arguments_str)
            else:
                arguments = arguments_str
        except json.JSONDecodeError as e:
            self.set_output_val(2, Data(f"Error parsing arguments: {e}"))
            self.exec_output(3)
            return

        # Execute the tool
        try:
            if callable(handler):
                result = handler(**arguments)
            else:
                result = f"Handler for {func_name} is not callable"

            # Create tool message for LLM
            from ..providers.base import Message
            tool_message = Message(
                role="tool",
                content=str(result),
                tool_call_id=tool_call_id,
            )

            self.set_output_val(0, Data(result))
            self.set_output_val(1, Data(tool_message))
            self.set_output_val(2, Data(None))

        except Exception as e:
            self.set_output_val(2, Data(f"Error executing tool: {e}"))

        self.exec_output(3)


class PythonToolNode(Node):
    """
    Create a tool from a Python function.

    Allows defining simple tools directly in the flow
    using Python code.

    Inputs:
        - name: Tool name
        - description: Tool description
        - code: Python code for the function body
        - parameters: Parameter definitions

    Outputs:
        - tool: Complete tool definition with handler
    """

    title = 'Python Tool'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='name'),
        NodeInputType(label='description'),
        NodeInputType(label='code'),
        NodeInputType(label='parameters'),
    ]
    init_outputs = [
        NodeOutputType(label='tool'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self._compiled_func = None

    def update_event(self, inp=-1):
        name_input = self.input(0)
        if not name_input or not name_input.payload:
            return

        name = str(name_input.payload)

        description = ""
        desc_input = self.input(1)
        if desc_input and desc_input.payload:
            description = str(desc_input.payload)

        code = "return None"
        code_input = self.input(2)
        if code_input and code_input.payload:
            code = str(code_input.payload)

        parameters = {"type": "object", "properties": {}}
        params_input = self.input(3)
        if params_input and params_input.payload:
            if isinstance(params_input.payload, dict):
                parameters = params_input.payload
            elif isinstance(params_input.payload, str):
                try:
                    parameters = json.loads(params_input.payload)
                except json.JSONDecodeError:
                    pass

        # Extract parameter names for function signature
        param_names = list(parameters.get("properties", {}).keys())
        param_str = ", ".join(param_names) if param_names else "**kwargs"

        # Create the function
        func_code = f"def {name}({param_str}):\n"
        for line in code.split("\n"):
            func_code += f"    {line}\n"

        try:
            exec_globals = {}
            exec(func_code, exec_globals)
            handler = exec_globals[name]
            self._compiled_func = handler
        except Exception as e:
            self.set_output_val(0, Data(f"Error compiling function: {e}"))
            return

        tool = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
            "_handler": handler,
        }

        self.set_output_val(0, Data(tool))


# Export all tool nodes
tool_nodes = [
    ToolDefinitionNode,
    ToolRegistryNode,
    ToolExecutorNode,
    PythonToolNode,
]
