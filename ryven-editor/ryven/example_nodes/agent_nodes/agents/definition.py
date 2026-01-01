"""
Agent definition and role nodes.

These nodes help users understand and configure agent roles,
responsibilities, and capabilities in a visual way.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from ryven.node_env import *


@dataclass
class AgentDefinition:
    """
    Complete definition of an agent.

    Captures the agent's identity, role, capabilities, and constraints.
    """
    name: str
    role: str
    persona: Optional[str] = None
    goal: Optional[str] = None
    backstory: Optional[str] = None

    # Capabilities and constraints
    tools: List[Dict[str, Any]] = field(default_factory=list)
    allowed_actions: List[str] = field(default_factory=list)
    forbidden_actions: List[str] = field(default_factory=list)

    # Behavioral settings
    verbose: bool = False
    allow_delegation: bool = False
    max_iterations: int = 10
    max_tokens_per_turn: int = 4096

    # Provider settings
    provider: Optional[Any] = None
    model: Optional[str] = None
    temperature: float = 0.7

    def to_system_prompt(self) -> str:
        """Generate a system prompt from this definition."""
        parts = []

        if self.role:
            parts.append(f"You are a {self.role}.")

        if self.persona:
            parts.append(f"\n{self.persona}")

        if self.goal:
            parts.append(f"\nYour goal: {self.goal}")

        if self.backstory:
            parts.append(f"\nBackground: {self.backstory}")

        if self.allowed_actions:
            parts.append(f"\nYou CAN: {', '.join(self.allowed_actions)}")

        if self.forbidden_actions:
            parts.append(f"\nYou MUST NOT: {', '.join(self.forbidden_actions)}")

        return "\n".join(parts)


class AgentNode(Node):
    """
    Define a complete agent with role, persona, and capabilities.

    This is the primary node for creating agents in the workflow.
    It combines role definition, persona, tools, and constraints
    into a single cohesive agent definition.

    Inputs:
        - name: Unique identifier for the agent
        - role: The agent's primary role/job title
        - persona: Detailed personality and behavior description
        - goal: What the agent is trying to achieve
        - tools: List of tools the agent can use
        - provider: LLM provider to use
        - model: Model to use for this agent

    Outputs:
        - agent: Complete AgentDefinition object
        - system_prompt: Generated system prompt for the agent
    """

    title = 'Agent'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='name'),
        NodeInputType(label='role'),
        NodeInputType(label='persona'),
        NodeInputType(label='goal'),
        NodeInputType(label='tools'),
        NodeInputType(label='provider'),
        NodeInputType(label='model'),
        NodeInputType(label='temperature', default=0.7),
    ]
    init_outputs = [
        NodeOutputType(label='agent'),
        NodeOutputType(label='system_prompt'),
    ]

    def update_event(self, inp=-1):
        name_input = self.input(0)
        if not name_input or not name_input.payload:
            return

        name = str(name_input.payload)

        role = ""
        role_input = self.input(1)
        if role_input and role_input.payload:
            role = str(role_input.payload)

        persona = None
        persona_input = self.input(2)
        if persona_input and persona_input.payload:
            persona = str(persona_input.payload)

        goal = None
        goal_input = self.input(3)
        if goal_input and goal_input.payload:
            goal = str(goal_input.payload)

        tools = []
        tools_input = self.input(4)
        if tools_input and tools_input.payload:
            tools = tools_input.payload if isinstance(tools_input.payload, list) else [tools_input.payload]

        provider = None
        provider_input = self.input(5)
        if provider_input and provider_input.payload:
            provider = provider_input.payload

        model = None
        model_input = self.input(6)
        if model_input and model_input.payload:
            model = str(model_input.payload)

        temperature = 0.7
        temp_input = self.input(7)
        if temp_input and temp_input.payload is not None:
            temperature = float(temp_input.payload)

        agent = AgentDefinition(
            name=name,
            role=role,
            persona=persona,
            goal=goal,
            tools=tools,
            provider=provider,
            model=model,
            temperature=temperature,
        )

        self.set_output_val(0, Data(agent))
        self.set_output_val(1, Data(agent.to_system_prompt()))


class RoleNode(Node):
    """
    Define an agent's role with clear responsibilities.

    Helps users think about what the agent should and shouldn't do.
    Inspired by CrewAI's role-based agent design.

    Inputs:
        - title: The role title (e.g., "Research Analyst")
        - responsibilities: What this role is responsible for
        - can_do: Explicit list of allowed actions
        - cannot_do: Explicit list of forbidden actions
        - reports_to: Role this agent reports to (for hierarchies)

    Outputs:
        - role: Complete role definition
        - summary: Human-readable role summary
    """

    title = 'Role'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='title'),
        NodeInputType(label='responsibilities'),
        NodeInputType(label='can_do'),
        NodeInputType(label='cannot_do'),
        NodeInputType(label='reports_to'),
    ]
    init_outputs = [
        NodeOutputType(label='role'),
        NodeOutputType(label='summary'),
    ]

    def update_event(self, inp=-1):
        title_input = self.input(0)
        if not title_input or not title_input.payload:
            return

        title = str(title_input.payload)

        responsibilities = []
        resp_input = self.input(1)
        if resp_input and resp_input.payload:
            if isinstance(resp_input.payload, list):
                responsibilities = resp_input.payload
            else:
                responsibilities = [str(resp_input.payload)]

        can_do = []
        can_input = self.input(2)
        if can_input and can_input.payload:
            if isinstance(can_input.payload, list):
                can_do = can_input.payload
            else:
                can_do = [str(can_input.payload)]

        cannot_do = []
        cannot_input = self.input(3)
        if cannot_input and cannot_input.payload:
            if isinstance(cannot_input.payload, list):
                cannot_do = cannot_input.payload
            else:
                cannot_do = [str(cannot_input.payload)]

        reports_to = None
        reports_input = self.input(4)
        if reports_input and reports_input.payload:
            reports_to = str(reports_input.payload)

        role = {
            'title': title,
            'responsibilities': responsibilities,
            'can_do': can_do,
            'cannot_do': cannot_do,
            'reports_to': reports_to,
        }

        # Create human-readable summary
        summary_parts = [f"Role: {title}"]
        if responsibilities:
            summary_parts.append(f"Responsible for: {', '.join(responsibilities)}")
        if can_do:
            summary_parts.append(f"Can: {', '.join(can_do)}")
        if cannot_do:
            summary_parts.append(f"Cannot: {', '.join(cannot_do)}")
        if reports_to:
            summary_parts.append(f"Reports to: {reports_to}")

        self.set_output_val(0, Data(role))
        self.set_output_val(1, Data("\n".join(summary_parts)))


class PersonaNode(Node):
    """
    Define an agent's personality and communication style.

    Helps create consistent agent behavior and tone.

    Inputs:
        - name: Character name
        - traits: List of personality traits
        - communication_style: How the agent communicates
        - expertise: Areas of expertise
        - background: Backstory or context

    Outputs:
        - persona: Complete persona definition
        - prompt_text: Text suitable for system prompt
    """

    title = 'Persona'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='name'),
        NodeInputType(label='traits'),
        NodeInputType(label='communication_style'),
        NodeInputType(label='expertise'),
        NodeInputType(label='background'),
    ]
    init_outputs = [
        NodeOutputType(label='persona'),
        NodeOutputType(label='prompt_text'),
    ]

    def update_event(self, inp=-1):
        name_input = self.input(0)
        if not name_input or not name_input.payload:
            return

        name = str(name_input.payload)

        traits = []
        traits_input = self.input(1)
        if traits_input and traits_input.payload:
            if isinstance(traits_input.payload, list):
                traits = traits_input.payload
            else:
                traits = [str(traits_input.payload)]

        communication_style = None
        style_input = self.input(2)
        if style_input and style_input.payload:
            communication_style = str(style_input.payload)

        expertise = []
        exp_input = self.input(3)
        if exp_input and exp_input.payload:
            if isinstance(exp_input.payload, list):
                expertise = exp_input.payload
            else:
                expertise = [str(exp_input.payload)]

        background = None
        bg_input = self.input(4)
        if bg_input and bg_input.payload:
            background = str(bg_input.payload)

        persona = {
            'name': name,
            'traits': traits,
            'communication_style': communication_style,
            'expertise': expertise,
            'background': background,
        }

        # Generate prompt text
        parts = [f"Your name is {name}."]
        if traits:
            parts.append(f"You are {', '.join(traits)}.")
        if communication_style:
            parts.append(f"You communicate in a {communication_style} manner.")
        if expertise:
            parts.append(f"You are an expert in: {', '.join(expertise)}.")
        if background:
            parts.append(f"Background: {background}")

        self.set_output_val(0, Data(persona))
        self.set_output_val(1, Data(" ".join(parts)))


class CapabilityNode(Node):
    """
    Define specific capabilities for an agent.

    Capabilities describe what an agent can do and any
    constraints on those abilities.

    Inputs:
        - name: Capability name
        - description: What this capability allows
        - requires_approval: Whether human approval is needed
        - max_uses: Maximum times this can be used (-1 for unlimited)
        - cooldown_seconds: Minimum time between uses

    Outputs:
        - capability: Capability definition
    """

    title = 'Capability'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='name'),
        NodeInputType(label='description'),
        NodeInputType(label='requires_approval', default=False),
        NodeInputType(label='max_uses', default=-1),
        NodeInputType(label='cooldown_seconds', default=0),
    ]
    init_outputs = [
        NodeOutputType(label='capability'),
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

        requires_approval = False
        approval_input = self.input(2)
        if approval_input and approval_input.payload is not None:
            requires_approval = bool(approval_input.payload)

        max_uses = -1
        uses_input = self.input(3)
        if uses_input and uses_input.payload is not None:
            max_uses = int(uses_input.payload)

        cooldown = 0
        cooldown_input = self.input(4)
        if cooldown_input and cooldown_input.payload is not None:
            cooldown = float(cooldown_input.payload)

        capability = {
            'name': name,
            'description': description,
            'requires_approval': requires_approval,
            'max_uses': max_uses,
            'cooldown_seconds': cooldown,
            'uses_remaining': max_uses if max_uses > 0 else None,
            'last_used': None,
        }

        self.set_output_val(0, Data(capability))


# Export all agent definition nodes
definition_nodes = [
    AgentNode,
    RoleNode,
    PersonaNode,
    CapabilityNode,
]
