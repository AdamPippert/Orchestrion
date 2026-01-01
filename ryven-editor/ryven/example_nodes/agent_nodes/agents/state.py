"""
Agent state and memory management nodes.

Provides nodes for managing agent state across turns
and conversations.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
from datetime import datetime
from ryven.node_env import *


class AgentStateNode(Node):
    """
    Manage persistent state for an agent.

    Tracks agent state across multiple interactions,
    including conversation history, task progress, and
    custom variables.

    Inputs:
        - agent: Agent definition
        - action: State action ('get', 'set', 'update', 'reset')
        - key: State key to access
        - value: Value to set (for set/update actions)
        - exec: Trigger execution

    Outputs:
        - state: Current state value
        - full_state: Complete agent state
        - done: Exec signal when complete
    """

    title = 'Agent State'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='agent'),
        NodeInputType(label='action'),
        NodeInputType(label='key'),
        NodeInputType(label='value'),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='state'),
        NodeOutputType(label='full_state'),
        NodeOutputType(type_='exec', label='done'),
    ]

    # Class-level state storage
    _agent_states: Dict[str, Dict[str, Any]] = {}

    def __init__(self, params):
        super().__init__(params)

    def _get_agent_name(self) -> Optional[str]:
        agent_input = self.input(0)
        if agent_input and agent_input.payload:
            agent = agent_input.payload
            if hasattr(agent, 'name'):
                return agent.name
            elif isinstance(agent, dict):
                return agent.get('name')
            else:
                return str(agent)
        return None

    def _ensure_state(self, agent_name: str) -> Dict[str, Any]:
        if agent_name not in self._agent_states:
            self._agent_states[agent_name] = {
                'created_at': datetime.now().isoformat(),
                'turn_count': 0,
                'variables': {},
            }
        return self._agent_states[agent_name]

    def update_event(self, inp=-1):
        if inp != 4:  # Only trigger on exec
            return

        agent_name = self._get_agent_name()
        if not agent_name:
            self.set_output_val(0, Data("Error: No agent provided"))
            self.exec_output(2)
            return

        action = 'get'
        action_input = self.input(1)
        if action_input and action_input.payload:
            action = str(action_input.payload).lower()

        key = None
        key_input = self.input(2)
        if key_input and key_input.payload:
            key = str(key_input.payload)

        value = None
        value_input = self.input(3)
        if value_input:
            value = value_input.payload

        state = self._ensure_state(agent_name)

        if action == 'get':
            if key:
                result = state.get('variables', {}).get(key)
            else:
                result = state
            self.set_output_val(0, Data(result))

        elif action == 'set':
            if key:
                state['variables'][key] = value
                self.set_output_val(0, Data(value))
            else:
                self.set_output_val(0, Data("Error: Key required for set"))

        elif action == 'update':
            if key and isinstance(value, dict):
                existing = state['variables'].get(key, {})
                if isinstance(existing, dict):
                    existing.update(value)
                    state['variables'][key] = existing
                else:
                    state['variables'][key] = value
                self.set_output_val(0, Data(state['variables'][key]))
            else:
                self.set_output_val(0, Data("Error: Key and dict value required for update"))

        elif action == 'reset':
            if agent_name in self._agent_states:
                del self._agent_states[agent_name]
            state = self._ensure_state(agent_name)
            self.set_output_val(0, Data(state))

        elif action == 'increment_turn':
            state['turn_count'] = state.get('turn_count', 0) + 1
            self.set_output_val(0, Data(state['turn_count']))

        self.set_output_val(1, Data(state))
        self.exec_output(2)


class AgentMemoryNode(Node):
    """
    Short-term memory for agent conversations.

    Maintains a sliding window of recent messages and
    can summarize older context.

    Inputs:
        - agent: Agent definition
        - message: New message to add
        - action: Memory action ('add', 'get_recent', 'clear', 'summarize')
        - window_size: Number of recent messages to keep
        - exec: Trigger execution

    Outputs:
        - messages: Recent messages in window
        - summary: Summary of older messages
        - total_count: Total messages ever added
        - done: Exec signal
    """

    title = 'Agent Memory'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='agent'),
        NodeInputType(label='message'),
        NodeInputType(label='action'),
        NodeInputType(label='window_size', default=10),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='messages'),
        NodeOutputType(label='summary'),
        NodeOutputType(label='total_count'),
        NodeOutputType(type_='exec', label='done'),
    ]

    # Class-level memory storage
    _agent_memories: Dict[str, Dict[str, Any]] = {}

    def _get_agent_name(self) -> Optional[str]:
        agent_input = self.input(0)
        if agent_input and agent_input.payload:
            agent = agent_input.payload
            if hasattr(agent, 'name'):
                return agent.name
            elif isinstance(agent, dict):
                return agent.get('name')
            else:
                return str(agent)
        return None

    def _ensure_memory(self, agent_name: str) -> Dict[str, Any]:
        if agent_name not in self._agent_memories:
            self._agent_memories[agent_name] = {
                'messages': [],
                'summary': None,
                'total_count': 0,
            }
        return self._agent_memories[agent_name]

    def update_event(self, inp=-1):
        if inp != 4:  # Only trigger on exec
            return

        agent_name = self._get_agent_name()
        if not agent_name:
            self.set_output_val(0, Data("Error: No agent provided"))
            self.exec_output(3)
            return

        action = 'get_recent'
        action_input = self.input(2)
        if action_input and action_input.payload:
            action = str(action_input.payload).lower()

        window_size = 10
        window_input = self.input(3)
        if window_input and window_input.payload is not None:
            window_size = int(window_input.payload)

        memory = self._ensure_memory(agent_name)

        if action == 'add':
            message_input = self.input(1)
            if message_input and message_input.payload:
                message = message_input.payload
                memory['messages'].append(message)
                memory['total_count'] += 1

        elif action == 'clear':
            memory['messages'] = []
            memory['summary'] = None

        elif action == 'get_all':
            pass  # Just return current state

        # Apply window
        messages = memory['messages']
        if len(messages) > window_size:
            recent = messages[-window_size:]
        else:
            recent = messages

        self.set_output_val(0, Data(recent))
        self.set_output_val(1, Data(memory.get('summary')))
        self.set_output_val(2, Data(memory['total_count']))
        self.exec_output(3)


class ConversationHistoryNode(Node):
    """
    Maintain conversation history for multi-turn chats.

    Automatically formats messages for LLM context and
    manages context window limits.

    Inputs:
        - message: New message to add (Message object or dict)
        - max_messages: Maximum messages to retain
        - max_tokens: Maximum tokens in history (approximate)
        - action: 'add', 'get', 'clear'
        - exec: Trigger

    Outputs:
        - history: List of messages for LLM
        - message_count: Number of messages
        - done: Exec signal
    """

    title = 'Conversation History'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='message'),
        NodeInputType(label='max_messages', default=50),
        NodeInputType(label='max_tokens'),
        NodeInputType(label='action'),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='history'),
        NodeOutputType(label='message_count'),
        NodeOutputType(type_='exec', label='done'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self._messages: List[Any] = []

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (4 chars per token)."""
        return len(text) // 4

    def update_event(self, inp=-1):
        if inp != 4:  # Only trigger on exec
            return

        action = 'get'
        action_input = self.input(3)
        if action_input and action_input.payload:
            action = str(action_input.payload).lower()

        max_messages = 50
        max_msg_input = self.input(1)
        if max_msg_input and max_msg_input.payload is not None:
            max_messages = int(max_msg_input.payload)

        max_tokens = None
        max_tok_input = self.input(2)
        if max_tok_input and max_tok_input.payload is not None:
            max_tokens = int(max_tok_input.payload)

        if action == 'add':
            message_input = self.input(0)
            if message_input and message_input.payload:
                self._messages.append(message_input.payload)

        elif action == 'clear':
            self._messages = []

        # Apply message limit
        if len(self._messages) > max_messages:
            self._messages = self._messages[-max_messages:]

        # Apply token limit if specified
        if max_tokens:
            total_tokens = 0
            kept_messages = []
            for msg in reversed(self._messages):
                content = ""
                if hasattr(msg, 'content'):
                    content = msg.content
                elif isinstance(msg, dict):
                    content = msg.get('content', '')

                if isinstance(content, str):
                    tokens = self._estimate_tokens(content)
                else:
                    tokens = 100  # Estimate for non-text content

                if total_tokens + tokens <= max_tokens:
                    kept_messages.insert(0, msg)
                    total_tokens += tokens
                else:
                    break

            self._messages = kept_messages

        self.set_output_val(0, Data(self._messages.copy()))
        self.set_output_val(1, Data(len(self._messages)))
        self.exec_output(2)

    def get_state(self) -> dict:
        return {'messages': self._messages}

    def set_state(self, data: dict, version):
        self._messages = data.get('messages', [])


# Export all state nodes
state_nodes = [
    AgentStateNode,
    AgentMemoryNode,
    ConversationHistoryNode,
]
