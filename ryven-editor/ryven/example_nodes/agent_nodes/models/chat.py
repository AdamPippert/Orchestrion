"""
Chat completion nodes for LLM interaction.
"""

from __future__ import annotations
import asyncio
from typing import List, Dict, Any, Optional
from ryven.node_env import *

from ..providers.base import (
    BaseProvider,
    Message,
    CompletionResponse,
    ProviderRegistry,
)


class ChatNodeBase(Node):
    """Base class for chat-related nodes."""
    version = 'v0.1'

    def get_provider(self) -> Optional[BaseProvider]:
        """Get provider from input or registry."""
        provider_input = self.input(0)
        if provider_input and provider_input.payload:
            return provider_input.payload
        # Fall back to default registered provider
        try:
            return ProviderRegistry.get()
        except ValueError:
            return None


class ChatNode(ChatNodeBase):
    """
    Send a chat completion request to an LLM.

    Inputs:
        - provider: The LLM provider to use (optional, uses default if not set)
        - messages: List of Message objects or dict format
        - temperature: Sampling temperature (0-2)
        - max_tokens: Maximum tokens to generate
        - tools: Tool definitions for function calling

    Outputs:
        - response: The model's text response
        - full_response: Complete CompletionResponse object with metadata
        - tool_calls: Any tool/function calls made by the model
    """

    title = 'Chat'
    init_inputs = [
        NodeInputType(label='provider'),
        NodeInputType(label='messages'),
        NodeInputType(label='temperature', default=0.7),
        NodeInputType(label='max_tokens'),
        NodeInputType(label='tools'),
        NodeInputType(label='model'),
    ]
    init_outputs = [
        NodeOutputType(label='response'),
        NodeOutputType(label='full_response'),
        NodeOutputType(label='tool_calls'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self._loop = None

    def _ensure_loop(self):
        """Get or create an event loop."""
        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop

    def update_event(self, inp=-1):
        provider = self.get_provider()
        if not provider:
            self.set_output_val(0, Data("Error: No provider configured"))
            return

        messages_input = self.input(1)
        if not messages_input or not messages_input.payload:
            return

        messages = messages_input.payload
        if not isinstance(messages, list):
            messages = [messages]

        # Convert to Message objects if needed
        msg_objects = []
        for m in messages:
            if isinstance(m, Message):
                msg_objects.append(m)
            elif isinstance(m, dict):
                msg_objects.append(Message(**m))

        # Get optional parameters
        temperature = 0.7
        temp_input = self.input(2)
        if temp_input and temp_input.payload is not None:
            temperature = float(temp_input.payload)

        max_tokens = None
        tokens_input = self.input(3)
        if tokens_input and tokens_input.payload:
            max_tokens = int(tokens_input.payload)

        tools = None
        tools_input = self.input(4)
        if tools_input and tools_input.payload:
            tools = tools_input.payload

        model = None
        model_input = self.input(5)
        if model_input and model_input.payload:
            model = model_input.payload

        # Run async chat
        loop = self._ensure_loop()

        async def do_chat():
            await provider.initialize()
            try:
                return await provider.chat(
                    messages=msg_objects,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                )
            finally:
                await provider.close()

        try:
            response = loop.run_until_complete(do_chat())
            self.set_output_val(0, Data(response.content))
            self.set_output_val(1, Data(response))
            self.set_output_val(2, Data(response.tool_calls))
        except Exception as e:
            self.set_output_val(0, Data(f"Error: {str(e)}"))


class ChatStreamNode(ChatNodeBase):
    """
    Stream a chat completion response from an LLM.

    Emits chunks as they arrive for real-time display.

    Inputs:
        - provider: The LLM provider to use
        - messages: List of messages
        - temperature: Sampling temperature
        - max_tokens: Maximum tokens

    Outputs:
        - chunk: Current text chunk (updates during streaming)
        - full_text: Complete accumulated text (after streaming ends)
        - done: Exec signal when streaming completes
    """

    title = 'Chat Stream'
    init_inputs = [
        NodeInputType(label='provider'),
        NodeInputType(label='messages'),
        NodeInputType(label='temperature', default=0.7),
        NodeInputType(label='max_tokens'),
        NodeInputType(label='model'),
        NodeInputType(type_='exec', label='start'),
    ]
    init_outputs = [
        NodeOutputType(label='chunk'),
        NodeOutputType(label='full_text'),
        NodeOutputType(type_='exec', label='on_chunk'),
        NodeOutputType(type_='exec', label='done'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self._accumulated = ""
        self._loop = None

    def _ensure_loop(self):
        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop

    def update_event(self, inp=-1):
        # Only trigger on exec input
        if inp != 5:
            return

        provider = self.get_provider()
        if not provider:
            return

        messages_input = self.input(1)
        if not messages_input or not messages_input.payload:
            return

        messages = messages_input.payload
        if not isinstance(messages, list):
            messages = [messages]

        msg_objects = []
        for m in messages:
            if isinstance(m, Message):
                msg_objects.append(m)
            elif isinstance(m, dict):
                msg_objects.append(Message(**m))

        temperature = 0.7
        temp_input = self.input(2)
        if temp_input and temp_input.payload is not None:
            temperature = float(temp_input.payload)

        max_tokens = None
        tokens_input = self.input(3)
        if tokens_input and tokens_input.payload:
            max_tokens = int(tokens_input.payload)

        model = None
        model_input = self.input(4)
        if model_input and model_input.payload:
            model = model_input.payload

        loop = self._ensure_loop()
        self._accumulated = ""

        async def do_stream():
            await provider.initialize()
            try:
                async for chunk in provider.chat_stream(
                    messages=msg_objects,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    self._accumulated += chunk.content
                    self.set_output_val(0, Data(chunk.content))
                    self.set_output_val(1, Data(self._accumulated))
                    self.exec_output(2)  # on_chunk
            finally:
                await provider.close()

        try:
            loop.run_until_complete(do_stream())
            self.exec_output(3)  # done
        except Exception as e:
            self.set_output_val(0, Data(f"Error: {str(e)}"))


class SystemPromptNode(Node):
    """
    Create a system message for chat context.

    The system message sets the behavior and personality of the AI.
    """

    title = 'System Prompt'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='prompt'),
    ]
    init_outputs = [
        NodeOutputType(label='message'),
    ]

    def update_event(self, inp=-1):
        prompt_input = self.input(0)
        if prompt_input and prompt_input.payload:
            message = Message(role="system", content=str(prompt_input.payload))
            self.set_output_val(0, Data(message))


class UserMessageNode(Node):
    """
    Create a user message.

    Represents input from the human user in the conversation.
    """

    title = 'User Message'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='content'),
        NodeInputType(label='name'),
    ]
    init_outputs = [
        NodeOutputType(label='message'),
    ]

    def update_event(self, inp=-1):
        content_input = self.input(0)
        if content_input and content_input.payload:
            name = None
            name_input = self.input(1)
            if name_input and name_input.payload:
                name = str(name_input.payload)

            message = Message(
                role="user",
                content=str(content_input.payload),
                name=name,
            )
            self.set_output_val(0, Data(message))


class AssistantMessageNode(Node):
    """
    Create an assistant message.

    Represents a response from the AI in the conversation.
    Useful for providing example responses or continuing conversations.
    """

    title = 'Assistant Message'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='content'),
        NodeInputType(label='tool_calls'),
    ]
    init_outputs = [
        NodeOutputType(label='message'),
    ]

    def update_event(self, inp=-1):
        content_input = self.input(0)
        if content_input and content_input.payload:
            tool_calls = None
            tc_input = self.input(1)
            if tc_input and tc_input.payload:
                tool_calls = tc_input.payload

            message = Message(
                role="assistant",
                content=str(content_input.payload),
                tool_calls=tool_calls,
            )
            self.set_output_val(0, Data(message))


class MessageListNode(Node):
    """
    Combine multiple messages into a conversation.

    Dynamically accepts any number of message inputs and
    outputs them as a list in order.
    """

    title = 'Message List'
    version = 'v0.1'
    init_inputs = []
    init_outputs = [
        NodeOutputType(label='messages'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self.num_inputs = 0

    def place_event(self):
        if self.num_inputs == 0:
            self.add_message_input()
            self.add_message_input()

    def add_message_input(self):
        self.create_input(label=f'msg_{self.num_inputs}')
        self.num_inputs += 1

    def remove_message_input(self, index):
        self.delete_input(index)
        self.num_inputs -= 1

    def update_event(self, inp=-1):
        messages = []
        for i in range(len(self.inputs)):
            msg_input = self.input(i)
            if msg_input and msg_input.payload:
                payload = msg_input.payload
                if isinstance(payload, Message):
                    messages.append(payload)
                elif isinstance(payload, dict):
                    messages.append(Message(**payload))
                elif isinstance(payload, list):
                    messages.extend(payload)

        self.set_output_val(0, Data(messages))

    def get_state(self) -> dict:
        return {'num_inputs': self.num_inputs}

    def set_state(self, data: dict, version):
        self.num_inputs = data.get('num_inputs', 0)


# Export all chat nodes
chat_nodes = [
    ChatNode,
    ChatStreamNode,
    SystemPromptNode,
    UserMessageNode,
    AssistantMessageNode,
    MessageListNode,
]
