"""
Tests for agent coordination patterns.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../ryven-editor/ryven/example_nodes'))


class TestSupervisorNode:
    """Tests for supervisor coordination pattern."""

    def test_supervisor_node_exists(self):
        """Verify SupervisorNode is properly defined."""
        from agent_nodes.agents.coordination import SupervisorNode

        assert hasattr(SupervisorNode, 'title')
        assert SupervisorNode.title == 'Supervisor'

    def test_worker_selection(self):
        """Test worker selection based on task."""
        from agent_nodes.agents.coordination import SupervisorNode
        from unittest.mock import MagicMock

        node = SupervisorNode.__new__(SupervisorNode)

        # Create mock workers
        worker1 = MagicMock()
        worker1.name = "researcher"
        worker1.role = "research analyst"

        worker2 = MagicMock()
        worker2.name = "writer"
        worker2.role = "content writer"

        workers = [worker1, worker2]

        # Task mentioning research should select researcher
        selected = node._select_worker(workers, "research the topic")
        assert selected == worker1

        # Task mentioning writing should select writer
        selected = node._select_worker(workers, "write an article")
        assert selected == worker2


class TestTeamNode:
    """Tests for team coordination pattern."""

    def test_team_node_exists(self):
        """Verify TeamNode is properly defined."""
        from agent_nodes.agents.coordination import TeamNode

        assert hasattr(TeamNode, 'title')
        assert TeamNode.title == 'Team'


class TestDebateNode:
    """Tests for debate pattern."""

    def test_debate_node_exists(self):
        """Verify DebateNode is properly defined."""
        from agent_nodes.agents.coordination import DebateNode

        assert hasattr(DebateNode, 'title')
        assert DebateNode.title == 'Debate'

    def test_debate_prompt_generation(self):
        """Test debate prompt generation."""
        from agent_nodes.agents.coordination import DebateNode

        node = DebateNode.__new__(DebateNode)

        prompt = node._generate_debate_prompt(
            agent_name="Alice",
            topic="AI safety",
            round_num=1,
            previous_arguments=[]
        )

        assert "Alice" in prompt
        assert "AI safety" in prompt
        assert "round 1" in prompt

    def test_debate_prompt_with_history(self):
        """Test debate prompt with previous arguments."""
        from agent_nodes.agents.coordination import DebateNode

        node = DebateNode.__new__(DebateNode)

        previous = [
            {'agent': 'Bob', 'argument': 'AI should be regulated'},
        ]

        prompt = node._generate_debate_prompt(
            agent_name="Alice",
            topic="AI regulation",
            round_num=2,
            previous_arguments=previous
        )

        assert "Bob" in prompt
        assert "regulated" in prompt


class TestConsensusNode:
    """Tests for consensus pattern."""

    def test_consensus_node_exists(self):
        """Verify ConsensusNode is properly defined."""
        from agent_nodes.agents.coordination import ConsensusNode

        assert hasattr(ConsensusNode, 'title')
        assert ConsensusNode.title == 'Consensus'

    def test_consensus_calculation(self):
        """Test consensus level calculation."""
        from agent_nodes.agents.coordination import ConsensusNode

        node = ConsensusNode.__new__(ConsensusNode)

        # All agree
        votes = {'a': True, 'b': True, 'c': True}
        assert node._calculate_consensus(votes) == 1.0

        # None agree
        votes = {'a': False, 'b': False, 'c': False}
        assert node._calculate_consensus(votes) == 0.0

        # Majority agrees
        votes = {'a': True, 'b': True, 'c': False}
        assert abs(node._calculate_consensus(votes) - 0.667) < 0.01

        # Empty votes
        votes = {}
        assert node._calculate_consensus(votes) == 0.0


class TestDelegatorNode:
    """Tests for delegation pattern."""

    def test_delegator_node_exists(self):
        """Verify DelegatorNode is properly defined."""
        from agent_nodes.agents.coordination import DelegatorNode

        assert hasattr(DelegatorNode, 'title')
        assert DelegatorNode.title == 'Delegator'

    def test_skill_matching(self):
        """Test skill matching score."""
        from agent_nodes.agents.coordination import DelegatorNode
        from unittest.mock import MagicMock

        node = DelegatorNode.__new__(DelegatorNode)

        agent = MagicMock()
        agent.role = "python developer"
        agent.tools = [{'name': 'code_review'}, {'name': 'testing'}]

        # Task matching role
        score = node._match_skill("need a developer", agent)
        assert score > 0

        # Task matching tool
        score = node._match_skill("do code_review", agent)
        assert score > 0

        # No match
        agent2 = MagicMock()
        agent2.role = "marketing"
        agent2.tools = []
        score = node._match_skill("write python code", agent2)
        assert score == 0


class TestHandoffNode:
    """Tests for handoff pattern."""

    def test_handoff_node_exists(self):
        """Verify HandoffNode is properly defined."""
        from agent_nodes.agents.coordination import HandoffNode

        assert hasattr(HandoffNode, 'title')
        assert HandoffNode.title == 'Handoff'

    def test_handoff_prompt_generation(self):
        """Test handoff prompt generation."""
        from agent_nodes.agents.coordination import HandoffNode

        node = HandoffNode.__new__(HandoffNode)

        prompt = node._generate_handoff_prompt(
            from_name="Agent A",
            to_name="Agent B",
            context={"task": "research", "progress": "50%"},
            reason="needs specialized skills"
        )

        assert "Agent A" in prompt
        assert "Agent B" in prompt
        assert "specialized skills" in prompt
        assert "task" in prompt or "research" in prompt


class TestMessageBusNode:
    """Tests for message bus pattern."""

    def test_message_bus_node_exists(self):
        """Verify MessageBusNode is properly defined."""
        from agent_nodes.agents.coordination import MessageBusNode

        assert hasattr(MessageBusNode, 'title')
        assert MessageBusNode.title == 'Message Bus'


class TestAgentMessage:
    """Tests for AgentMessage dataclass."""

    def test_agent_message_creation(self):
        """Test creating an agent message."""
        from agent_nodes.agents.coordination import AgentMessage

        msg = AgentMessage(
            sender="agent1",
            recipient="agent2",
            content="Hello!",
            message_type="greeting"
        )

        assert msg.sender == "agent1"
        assert msg.recipient == "agent2"
        assert msg.content == "Hello!"
        assert msg.message_type == "greeting"
        assert msg.timestamp > 0


class TestTaskAssignment:
    """Tests for TaskAssignment dataclass."""

    def test_task_assignment_creation(self):
        """Test creating a task assignment."""
        from agent_nodes.agents.coordination import TaskAssignment

        task = TaskAssignment(
            task_id="task-1",
            description="Complete analysis",
            assigned_to="analyst",
            assigned_by="supervisor",
            priority=8
        )

        assert task.task_id == "task-1"
        assert task.assigned_to == "analyst"
        assert task.priority == 8
        assert task.status == "pending"
