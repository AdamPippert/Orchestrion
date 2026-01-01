"""
Agent definition nodes.

Provides nodes for defining:
- Agent roles and personas
- Tool/capability registration
- Agent state management
- Multi-agent coordination
- Reasoning patterns (ReAct, CoT, etc.)
"""

from .definition import (
    AgentNode,
    RoleNode,
    PersonaNode,
    CapabilityNode,
)
from .tools import (
    ToolDefinitionNode,
    ToolRegistryNode,
    ToolExecutorNode,
    PythonToolNode,
)
from .state import (
    AgentStateNode,
    AgentMemoryNode,
)
from .coordination import (
    SupervisorNode,
    TeamNode,
    DebateNode,
    ConsensusNode,
    DelegatorNode,
    HandoffNode,
    MessageBusNode,
)
from .reasoning import (
    ReActNode,
    PlanExecuteNode,
    ChainOfThoughtNode,
    ReflectionNode,
    TreeOfThoughtsNode,
    SelfAskNode,
)

__all__ = [
    # Definition
    'AgentNode',
    'RoleNode',
    'PersonaNode',
    'CapabilityNode',
    # Tools
    'ToolDefinitionNode',
    'ToolRegistryNode',
    'ToolExecutorNode',
    'PythonToolNode',
    # State
    'AgentStateNode',
    'AgentMemoryNode',
    # Coordination
    'SupervisorNode',
    'TeamNode',
    'DebateNode',
    'ConsensusNode',
    'DelegatorNode',
    'HandoffNode',
    'MessageBusNode',
    # Reasoning
    'ReActNode',
    'PlanExecuteNode',
    'ChainOfThoughtNode',
    'ReflectionNode',
    'TreeOfThoughtsNode',
    'SelfAskNode',
]
