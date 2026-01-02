"""
Multi-agent coordination patterns.

Implements various coordination strategies for multi-agent systems:
- Supervisor pattern (hierarchical coordination)
- Team coordination
- Debate and deliberation
- Consensus building
- Task delegation
- Agent handoff
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import time
import uuid
from ryven.node_env import *


class CoordinationStrategy(Enum):
    """Available coordination strategies."""
    SUPERVISOR = "supervisor"      # One agent supervises others
    ROUND_ROBIN = "round_robin"    # Agents take turns
    PARALLEL = "parallel"          # All agents work simultaneously
    HIERARCHICAL = "hierarchical"  # Tree structure of agents
    CONSENSUS = "consensus"        # Agents must agree
    DEBATE = "debate"              # Agents argue/discuss


@dataclass
class AgentMessage:
    """Message passed between agents."""
    sender: str
    recipient: str  # Use "*" for broadcast
    content: Any
    message_type: str = "message"  # message, task, result, feedback
    timestamp: float = field(default_factory=time.time)
    conversation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskAssignment:
    """A task assigned to an agent."""
    task_id: str
    description: str
    assigned_to: str
    assigned_by: str
    priority: int = 5  # 1-10
    deadline: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[Any] = None
    created_at: float = field(default_factory=time.time)


class SupervisorNode(Node):
    """
    Supervisor pattern for hierarchical agent coordination.

    A supervisor agent oversees worker agents, delegating tasks,
    reviewing results, and making final decisions.

    Inputs:
        - supervisor: The supervising agent definition
        - workers: List of worker agent definitions
        - task: The task to complete
        - max_iterations: Maximum delegation rounds
        - require_approval: Whether supervisor must approve results

    Outputs:
        - result: Final result after supervision
        - delegation_log: History of delegations and reviews
        - metrics: Supervision metrics
    """

    title = 'Supervisor'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='supervisor'),
        NodeInputType(label='workers'),
        NodeInputType(label='task'),
        NodeInputType(label='max_iterations', default=5),
        NodeInputType(label='require_approval', default=True),
    ]
    init_outputs = [
        NodeOutputType(label='result'),
        NodeOutputType(label='delegation_log'),
        NodeOutputType(label='metrics'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self.delegation_log = []
        self.iteration = 0

    def _select_worker(self, workers: List[Any], task: str) -> Any:
        """Select the best worker for a task based on capabilities."""
        if not workers:
            return None

        # Simple selection - in practice, use semantic matching
        # Check for role/capability match
        for worker in workers:
            if hasattr(worker, 'role') and worker.role:
                if any(keyword in task.lower() for keyword in worker.role.lower().split()):
                    return worker

        # Default to first worker
        return workers[0]

    def _create_delegation(
        self,
        supervisor: Any,
        worker: Any,
        task: str
    ) -> TaskAssignment:
        """Create a task delegation."""
        return TaskAssignment(
            task_id=str(uuid.uuid4())[:8],
            description=task,
            assigned_to=getattr(worker, 'name', 'worker'),
            assigned_by=getattr(supervisor, 'name', 'supervisor'),
            priority=5,
        )

    def update_event(self, inp=-1):
        supervisor_input = self.input(0)
        if not supervisor_input or not supervisor_input.payload:
            return

        supervisor = supervisor_input.payload

        workers = []
        workers_input = self.input(1)
        if workers_input and workers_input.payload:
            if isinstance(workers_input.payload, list):
                workers = workers_input.payload
            else:
                workers = [workers_input.payload]

        task = ""
        task_input = self.input(2)
        if task_input and task_input.payload:
            task = str(task_input.payload)

        max_iter = 5
        iter_input = self.input(3)
        if iter_input and iter_input.payload is not None:
            max_iter = int(iter_input.payload)

        require_approval = True
        approval_input = self.input(4)
        if approval_input and approval_input.payload is not None:
            require_approval = bool(approval_input.payload)

        # Select worker and create delegation
        selected_worker = self._select_worker(workers, task)
        if selected_worker:
            delegation = self._create_delegation(supervisor, selected_worker, task)
            self.delegation_log.append({
                'action': 'delegate',
                'task': delegation.description,
                'worker': delegation.assigned_to,
                'timestamp': time.time(),
            })

        # Output structure for workflow to use
        result = {
            'supervisor': getattr(supervisor, 'name', 'supervisor'),
            'workers': [getattr(w, 'name', f'worker_{i}') for i, w in enumerate(workers)],
            'task': task,
            'selected_worker': getattr(selected_worker, 'name', 'worker') if selected_worker else None,
            'require_approval': require_approval,
            'max_iterations': max_iter,
        }

        metrics = {
            'delegations': len(self.delegation_log),
            'workers_available': len(workers),
        }

        self.set_output_val(0, Data(result))
        self.set_output_val(1, Data(self.delegation_log))
        self.set_output_val(2, Data(metrics))


class TeamNode(Node):
    """
    Team coordination for collaborative multi-agent work.

    Creates a team of agents with shared goals and coordination.
    Agents can communicate, share context, and work together.

    Inputs:
        - agents: List of agent definitions
        - team_goal: Shared objective for the team
        - coordination_strategy: How agents coordinate
        - shared_context: Initial shared context

    Outputs:
        - team: Configured team object
        - communication_channel: Channel for inter-agent messages
    """

    title = 'Team'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='agents'),
        NodeInputType(label='team_goal'),
        NodeInputType(label='coordination_strategy', default='parallel'),
        NodeInputType(label='shared_context'),
    ]
    init_outputs = [
        NodeOutputType(label='team'),
        NodeOutputType(label='communication_channel'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self.message_queue: List[AgentMessage] = []

    def update_event(self, inp=-1):
        agents = []
        agents_input = self.input(0)
        if agents_input and agents_input.payload:
            if isinstance(agents_input.payload, list):
                agents = agents_input.payload
            else:
                agents = [agents_input.payload]

        team_goal = ""
        goal_input = self.input(1)
        if goal_input and goal_input.payload:
            team_goal = str(goal_input.payload)

        strategy = "parallel"
        strategy_input = self.input(2)
        if strategy_input and strategy_input.payload:
            strategy = str(strategy_input.payload)

        shared_context = {}
        context_input = self.input(3)
        if context_input and context_input.payload:
            shared_context = context_input.payload if isinstance(context_input.payload, dict) else {}

        team = {
            'id': str(uuid.uuid4())[:8],
            'agents': [getattr(a, 'name', f'agent_{i}') for i, a in enumerate(agents)],
            'agent_definitions': agents,
            'goal': team_goal,
            'strategy': strategy,
            'shared_context': shared_context,
            'created_at': time.time(),
        }

        # Communication channel for message passing
        channel = {
            'team_id': team['id'],
            'message_queue': self.message_queue,
            'send': lambda msg: self.message_queue.append(msg),
            'broadcast': lambda content, sender: self.message_queue.append(
                AgentMessage(sender=sender, recipient="*", content=content)
            ),
        }

        self.set_output_val(0, Data(team))
        self.set_output_val(1, Data(channel))


class DebateNode(Node):
    """
    Debate/deliberation pattern for agent decision-making.

    Multiple agents present arguments and counter-arguments
    to reach better decisions through structured debate.

    Inputs:
        - agents: Debating agents (min 2)
        - topic: The topic to debate
        - rounds: Number of debate rounds
        - judge: Optional judge agent for final decision

    Outputs:
        - debate_transcript: Full debate history
        - conclusion: Final position/decision
        - positions: Each agent's final position
    """

    title = 'Debate'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='agents'),
        NodeInputType(label='topic'),
        NodeInputType(label='rounds', default=3),
        NodeInputType(label='judge'),
    ]
    init_outputs = [
        NodeOutputType(label='debate_transcript'),
        NodeOutputType(label='conclusion'),
        NodeOutputType(label='positions'),
    ]

    def _generate_debate_prompt(
        self,
        agent_name: str,
        topic: str,
        round_num: int,
        previous_arguments: List[Dict]
    ) -> str:
        """Generate a debate prompt for an agent."""
        prompt = f"You are {agent_name} in a structured debate.\n\n"
        prompt += f"Topic: {topic}\n\n"

        if previous_arguments:
            prompt += "Previous arguments:\n"
            for arg in previous_arguments:
                prompt += f"- {arg['agent']}: {arg['argument']}\n"
            prompt += "\n"

        prompt += f"This is round {round_num}. Present your argument or counter-argument."
        return prompt

    def update_event(self, inp=-1):
        agents = []
        agents_input = self.input(0)
        if agents_input and agents_input.payload:
            if isinstance(agents_input.payload, list):
                agents = agents_input.payload
            else:
                agents = [agents_input.payload]

        topic = ""
        topic_input = self.input(1)
        if topic_input and topic_input.payload:
            topic = str(topic_input.payload)

        rounds = 3
        rounds_input = self.input(2)
        if rounds_input and rounds_input.payload is not None:
            rounds = int(rounds_input.payload)

        judge = None
        judge_input = self.input(3)
        if judge_input and judge_input.payload:
            judge = judge_input.payload

        # Set up debate structure
        debate_config = {
            'topic': topic,
            'agents': [getattr(a, 'name', f'debater_{i}') for i, a in enumerate(agents)],
            'agent_definitions': agents,
            'rounds': rounds,
            'judge': getattr(judge, 'name', None) if judge else None,
            'judge_definition': judge,
            'transcript': [],
            'positions': {getattr(a, 'name', f'debater_{i}'): None for i, a in enumerate(agents)},
        }

        # Generate initial prompts for each agent
        for i, agent in enumerate(agents):
            agent_name = getattr(agent, 'name', f'debater_{i}')
            debate_config['positions'][agent_name] = {
                'initial_prompt': self._generate_debate_prompt(
                    agent_name, topic, 1, []
                ),
                'role': 'debater',
            }

        self.set_output_val(0, Data(debate_config))
        self.set_output_val(1, Data(None))  # Conclusion comes after debate runs
        self.set_output_val(2, Data(debate_config['positions']))


class ConsensusNode(Node):
    """
    Consensus-building pattern for multi-agent agreement.

    Agents work to reach agreement on a decision or output.
    Supports various consensus mechanisms.

    Inputs:
        - agents: Participating agents
        - proposal: Initial proposal to consider
        - consensus_threshold: Required agreement level (0-1)
        - max_rounds: Maximum negotiation rounds

    Outputs:
        - consensus_reached: Whether consensus was achieved
        - final_decision: The agreed-upon decision
        - votes: How each agent voted
        - discussion: Negotiation history
    """

    title = 'Consensus'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='agents'),
        NodeInputType(label='proposal'),
        NodeInputType(label='consensus_threshold', default=0.75),
        NodeInputType(label='max_rounds', default=5),
    ]
    init_outputs = [
        NodeOutputType(label='consensus_reached'),
        NodeOutputType(label='final_decision'),
        NodeOutputType(label='votes'),
        NodeOutputType(label='discussion'),
    ]

    def _calculate_consensus(self, votes: Dict[str, bool]) -> float:
        """Calculate consensus level from votes."""
        if not votes:
            return 0.0
        agree_count = sum(1 for v in votes.values() if v)
        return agree_count / len(votes)

    def update_event(self, inp=-1):
        agents = []
        agents_input = self.input(0)
        if agents_input and agents_input.payload:
            if isinstance(agents_input.payload, list):
                agents = agents_input.payload
            else:
                agents = [agents_input.payload]

        proposal = ""
        proposal_input = self.input(1)
        if proposal_input and proposal_input.payload:
            proposal = str(proposal_input.payload)

        threshold = 0.75
        threshold_input = self.input(2)
        if threshold_input and threshold_input.payload is not None:
            threshold = float(threshold_input.payload)

        max_rounds = 5
        rounds_input = self.input(3)
        if rounds_input and rounds_input.payload is not None:
            max_rounds = int(rounds_input.payload)

        # Set up consensus structure
        agent_names = [getattr(a, 'name', f'agent_{i}') for i, a in enumerate(agents)]

        consensus_config = {
            'proposal': proposal,
            'agents': agent_names,
            'agent_definitions': agents,
            'threshold': threshold,
            'max_rounds': max_rounds,
            'votes': {name: None for name in agent_names},
            'discussion': [],
            'current_round': 0,
        }

        # Initial state
        self.set_output_val(0, Data(False))  # Not yet reached
        self.set_output_val(1, Data(consensus_config))
        self.set_output_val(2, Data(consensus_config['votes']))
        self.set_output_val(3, Data(consensus_config['discussion']))


class DelegatorNode(Node):
    """
    Task delegation pattern based on agent capabilities.

    Analyzes tasks and delegates to the most appropriate agent
    based on skills, availability, and workload.

    Inputs:
        - agents: Available agents with capabilities
        - tasks: Tasks to delegate
        - strategy: Delegation strategy (skill_match, load_balance, round_robin)
        - max_tasks_per_agent: Maximum concurrent tasks per agent

    Outputs:
        - assignments: Task-to-agent assignments
        - unassigned: Tasks that couldn't be assigned
        - workload: Current workload per agent
    """

    title = 'Delegator'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='agents'),
        NodeInputType(label='tasks'),
        NodeInputType(label='strategy', default='skill_match'),
        NodeInputType(label='max_tasks_per_agent', default=3),
    ]
    init_outputs = [
        NodeOutputType(label='assignments'),
        NodeOutputType(label='unassigned'),
        NodeOutputType(label='workload'),
    ]

    def _match_skill(self, task: str, agent: Any) -> float:
        """Calculate skill match score between task and agent."""
        score = 0.0

        # Check agent role
        if hasattr(agent, 'role') and agent.role:
            role_words = set(agent.role.lower().split())
            task_words = set(task.lower().split())
            overlap = len(role_words & task_words)
            score += overlap * 0.3

        # Check capabilities
        if hasattr(agent, 'tools') and agent.tools:
            for tool in agent.tools:
                tool_name = tool.get('name', '') if isinstance(tool, dict) else str(tool)
                if tool_name.lower() in task.lower():
                    score += 0.5

        return min(score, 1.0)

    def update_event(self, inp=-1):
        agents = []
        agents_input = self.input(0)
        if agents_input and agents_input.payload:
            if isinstance(agents_input.payload, list):
                agents = agents_input.payload
            else:
                agents = [agents_input.payload]

        tasks = []
        tasks_input = self.input(1)
        if tasks_input and tasks_input.payload:
            if isinstance(tasks_input.payload, list):
                tasks = tasks_input.payload
            else:
                tasks = [str(tasks_input.payload)]

        strategy = "skill_match"
        strategy_input = self.input(2)
        if strategy_input and strategy_input.payload:
            strategy = str(strategy_input.payload)

        max_tasks = 3
        max_input = self.input(3)
        if max_input and max_input.payload is not None:
            max_tasks = int(max_input.payload)

        # Track assignments and workload
        assignments = []
        unassigned = []
        workload = {getattr(a, 'name', f'agent_{i}'): 0 for i, a in enumerate(agents)}

        for task in tasks:
            task_str = str(task)
            best_agent = None
            best_score = -1

            if strategy == "skill_match":
                for agent in agents:
                    agent_name = getattr(agent, 'name', 'agent')
                    if workload.get(agent_name, 0) < max_tasks:
                        score = self._match_skill(task_str, agent)
                        if score > best_score:
                            best_score = score
                            best_agent = agent

            elif strategy == "load_balance":
                # Find agent with least work
                min_load = float('inf')
                for agent in agents:
                    agent_name = getattr(agent, 'name', 'agent')
                    load = workload.get(agent_name, 0)
                    if load < min_load and load < max_tasks:
                        min_load = load
                        best_agent = agent

            elif strategy == "round_robin":
                # Simple round robin
                for agent in agents:
                    agent_name = getattr(agent, 'name', 'agent')
                    if workload.get(agent_name, 0) < max_tasks:
                        best_agent = agent
                        break

            if best_agent:
                agent_name = getattr(best_agent, 'name', 'agent')
                assignments.append({
                    'task': task_str,
                    'agent': agent_name,
                    'score': best_score if strategy == "skill_match" else None,
                })
                workload[agent_name] = workload.get(agent_name, 0) + 1
            else:
                unassigned.append(task_str)

        self.set_output_val(0, Data(assignments))
        self.set_output_val(1, Data(unassigned))
        self.set_output_val(2, Data(workload))


class HandoffNode(Node):
    """
    Agent handoff pattern for transferring work between agents.

    Supports clean handoff of context, state, and responsibilities
    from one agent to another.

    Inputs:
        - from_agent: Agent handing off
        - to_agent: Agent receiving
        - context: Current context to transfer
        - reason: Reason for handoff
        - preserve_history: Whether to include conversation history

    Outputs:
        - handoff: Complete handoff package
        - handoff_prompt: Prompt for receiving agent
        - summary: Handoff summary for logging
    """

    title = 'Handoff'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='from_agent'),
        NodeInputType(label='to_agent'),
        NodeInputType(label='context'),
        NodeInputType(label='reason'),
        NodeInputType(label='preserve_history', default=True),
    ]
    init_outputs = [
        NodeOutputType(label='handoff'),
        NodeOutputType(label='handoff_prompt'),
        NodeOutputType(label='summary'),
    ]

    def _generate_handoff_prompt(
        self,
        from_name: str,
        to_name: str,
        context: Any,
        reason: str,
    ) -> str:
        """Generate a handoff prompt for the receiving agent."""
        prompt = f"You are receiving a handoff from {from_name}.\n\n"
        prompt += f"Reason for handoff: {reason}\n\n"

        if context:
            prompt += "Context from previous agent:\n"
            if isinstance(context, dict):
                for key, value in context.items():
                    prompt += f"- {key}: {value}\n"
            else:
                prompt += str(context)
            prompt += "\n"

        prompt += "Please continue from where the previous agent left off."
        return prompt

    def update_event(self, inp=-1):
        from_agent = None
        from_input = self.input(0)
        if from_input and from_input.payload:
            from_agent = from_input.payload

        to_agent = None
        to_input = self.input(1)
        if to_input and to_input.payload:
            to_agent = to_input.payload

        context = {}
        context_input = self.input(2)
        if context_input and context_input.payload:
            context = context_input.payload

        reason = ""
        reason_input = self.input(3)
        if reason_input and reason_input.payload:
            reason = str(reason_input.payload)

        preserve_history = True
        history_input = self.input(4)
        if history_input and history_input.payload is not None:
            preserve_history = bool(history_input.payload)

        from_name = getattr(from_agent, 'name', 'previous_agent') if from_agent else 'previous_agent'
        to_name = getattr(to_agent, 'name', 'receiving_agent') if to_agent else 'receiving_agent'

        handoff = {
            'id': str(uuid.uuid4())[:8],
            'from_agent': from_name,
            'to_agent': to_name,
            'from_definition': from_agent,
            'to_definition': to_agent,
            'context': context,
            'reason': reason,
            'preserve_history': preserve_history,
            'timestamp': time.time(),
        }

        handoff_prompt = self._generate_handoff_prompt(
            from_name, to_name, context, reason
        )

        summary = f"Handoff from {from_name} to {to_name}: {reason}"

        self.set_output_val(0, Data(handoff))
        self.set_output_val(1, Data(handoff_prompt))
        self.set_output_val(2, Data(summary))


class MessageBusNode(Node):
    """
    Message bus for inter-agent communication.

    Provides a publish-subscribe pattern for agent communication
    with topic-based routing.

    Inputs:
        - action: send, subscribe, or broadcast
        - topic: Message topic/channel
        - message: Message content
        - sender: Sender agent name
        - filter: Optional message filter

    Outputs:
        - result: Operation result
        - messages: Received messages (for subscribe)
    """

    title = 'Message Bus'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='action', default='send'),
        NodeInputType(label='topic'),
        NodeInputType(label='message'),
        NodeInputType(label='sender'),
        NodeInputType(label='filter'),
    ]
    init_outputs = [
        NodeOutputType(label='result'),
        NodeOutputType(label='messages'),
    ]

    # Class-level message storage (shared across instances)
    _topics: Dict[str, List[AgentMessage]] = {}
    _subscriptions: Dict[str, List[str]] = {}

    def update_event(self, inp=-1):
        action = "send"
        action_input = self.input(0)
        if action_input and action_input.payload:
            action = str(action_input.payload)

        topic = "default"
        topic_input = self.input(1)
        if topic_input and topic_input.payload:
            topic = str(topic_input.payload)

        message = None
        message_input = self.input(2)
        if message_input and message_input.payload:
            message = message_input.payload

        sender = "anonymous"
        sender_input = self.input(3)
        if sender_input and sender_input.payload:
            sender = str(sender_input.payload)

        filter_func = None
        filter_input = self.input(4)
        if filter_input and filter_input.payload:
            filter_func = filter_input.payload

        if action == "send":
            # Send message to topic
            if topic not in self._topics:
                self._topics[topic] = []

            agent_msg = AgentMessage(
                sender=sender,
                recipient=topic,
                content=message,
                message_type="topic_message",
            )
            self._topics[topic].append(agent_msg)

            self.set_output_val(0, Data({'status': 'sent', 'topic': topic}))
            self.set_output_val(1, Data([]))

        elif action == "subscribe":
            # Get messages from topic
            messages = self._topics.get(topic, [])

            if filter_func and callable(filter_func):
                messages = [m for m in messages if filter_func(m)]

            self.set_output_val(0, Data({'status': 'subscribed', 'topic': topic}))
            self.set_output_val(1, Data(messages))

        elif action == "broadcast":
            # Send to all topics
            agent_msg = AgentMessage(
                sender=sender,
                recipient="*",
                content=message,
                message_type="broadcast",
            )

            for topic_name in self._topics:
                self._topics[topic_name].append(agent_msg)

            self.set_output_val(0, Data({'status': 'broadcast', 'topics': list(self._topics.keys())}))
            self.set_output_val(1, Data([]))


# Export all coordination nodes
coordination_nodes = [
    SupervisorNode,
    TeamNode,
    DebateNode,
    ConsensusNode,
    DelegatorNode,
    HandoffNode,
    MessageBusNode,
]
