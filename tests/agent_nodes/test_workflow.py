"""
Tests for workflow composition and chain patterns.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../ryven-editor/ryven/example_nodes'))


class TestLLMChainNode:
    """Tests for LLM chain pattern."""

    def test_llm_chain_node_exists(self):
        """Verify LLMChainNode is properly defined."""
        from agent_nodes.routing.workflow import LLMChainNode

        assert hasattr(LLMChainNode, 'title')
        assert LLMChainNode.title == 'LLM Chain'

    def test_template_application(self):
        """Test chain template application."""
        from agent_nodes.routing.workflow import LLMChainNode

        node = LLMChainNode.__new__(LLMChainNode)
        node.intermediate_outputs = []
        node.current_step = 0

        # Empty outputs
        result = node._apply_template("Start: {previous_output}", [], 0)
        assert result == "Start: "

        # With previous output
        result = node._apply_template(
            "Continue from: {previous_output}",
            ["First response"],
            1
        )
        assert "First response" in result

        # All outputs
        result = node._apply_template(
            "All: {all_outputs}",
            ["Response 1", "Response 2"],
            2
        )
        assert "Response 1" in result
        assert "Response 2" in result

        # Step number
        result = node._apply_template(
            "Step {step}",
            [],
            5
        )
        assert "5" in result


class TestTransformChainNode:
    """Tests for transform chain pattern."""

    def test_transform_chain_node_exists(self):
        """Verify TransformChainNode is properly defined."""
        from agent_nodes.routing.workflow import TransformChainNode

        assert hasattr(TransformChainNode, 'title')
        assert TransformChainNode.title == 'Transform Chain'

    def test_apply_transform_none(self):
        """Test transform with no transformation."""
        from agent_nodes.routing.workflow import TransformChainNode

        node = TransformChainNode.__new__(TransformChainNode)

        result = node._apply_transform("hello", None)
        assert result == "hello"

    def test_apply_transform_callable(self):
        """Test transform with callable."""
        from agent_nodes.routing.workflow import TransformChainNode

        node = TransformChainNode.__new__(TransformChainNode)

        result = node._apply_transform("hello", str.upper)
        assert result == "HELLO"

    def test_apply_transform_template(self):
        """Test transform with template string."""
        from agent_nodes.routing.workflow import TransformChainNode

        node = TransformChainNode.__new__(TransformChainNode)

        result = node._apply_transform("world", "Hello {input}!")
        assert result == "Hello world!"


class TestRouterChainNode:
    """Tests for router chain pattern."""

    def test_router_chain_node_exists(self):
        """Verify RouterChainNode is properly defined."""
        from agent_nodes.routing.workflow import RouterChainNode

        assert hasattr(RouterChainNode, 'title')
        assert RouterChainNode.title == 'Router Chain'

    def test_classify_input_default(self):
        """Test classification with no classifier."""
        from agent_nodes.routing.workflow import RouterChainNode

        node = RouterChainNode.__new__(RouterChainNode)

        result = node._classify_input("any data", None)
        assert result == "default"

    def test_classify_input_callable(self):
        """Test classification with callable."""
        from agent_nodes.routing.workflow import RouterChainNode

        node = RouterChainNode.__new__(RouterChainNode)

        classifier = lambda x: "urgent" if "urgent" in x.lower() else "normal"
        result = node._classify_input("URGENT: Please help", classifier)
        assert result == "urgent"

    def test_classify_input_keyword_dict(self):
        """Test classification with keyword dict."""
        from agent_nodes.routing.workflow import RouterChainNode

        node = RouterChainNode.__new__(RouterChainNode)

        classifier = {
            "help": "support",
            "buy": "sales",
            "bug": "technical",
        }

        result = node._classify_input("I need help with my account", classifier)
        assert result == "support"

        result = node._classify_input("I want to buy something", classifier)
        assert result == "sales"


class TestWorkflowNode:
    """Tests for workflow definition."""

    def test_workflow_node_exists(self):
        """Verify WorkflowNode is properly defined."""
        from agent_nodes.routing.workflow import WorkflowNode

        assert hasattr(WorkflowNode, 'title')
        assert WorkflowNode.title == 'Workflow'


class TestWorkflowStepNode:
    """Tests for workflow step definition."""

    def test_workflow_step_node_exists(self):
        """Verify WorkflowStepNode is properly defined."""
        from agent_nodes.routing.workflow import WorkflowStepNode

        assert hasattr(WorkflowStepNode, 'title')
        assert WorkflowStepNode.title == 'Workflow Step'


class TestWorkflowRunnerNode:
    """Tests for workflow execution."""

    def test_workflow_runner_node_exists(self):
        """Verify WorkflowRunnerNode is properly defined."""
        from agent_nodes.routing.workflow import WorkflowRunnerNode

        assert hasattr(WorkflowRunnerNode, 'title')
        assert WorkflowRunnerNode.title == 'Workflow Runner'


class TestPipelineNode:
    """Tests for data pipeline."""

    def test_pipeline_node_exists(self):
        """Verify PipelineNode is properly defined."""
        from agent_nodes.routing.workflow import PipelineNode

        assert hasattr(PipelineNode, 'title')
        assert PipelineNode.title == 'Pipeline'


class TestStateMachineNode:
    """Tests for state machine."""

    def test_state_machine_node_exists(self):
        """Verify StateMachineNode is properly defined."""
        from agent_nodes.routing.workflow import StateMachineNode

        assert hasattr(StateMachineNode, 'title')
        assert StateMachineNode.title == 'State Machine'

    def test_get_available_transitions(self):
        """Test getting available transitions."""
        from agent_nodes.routing.workflow import StateMachineNode

        node = StateMachineNode.__new__(StateMachineNode)
        node.current_state = None
        node.history = []

        transitions = [
            {'from': 'idle', 'to': 'running', 'on': 'start'},
            {'from': 'idle', 'to': 'configuring', 'on': 'configure'},
            {'from': 'running', 'to': 'idle', 'on': 'stop'},
            {'from': 'running', 'to': 'paused', 'on': 'pause'},
        ]

        # From idle state
        available = node._get_available_transitions('idle', transitions)
        assert len(available) == 2

        # From running state
        available = node._get_available_transitions('running', transitions)
        assert len(available) == 2

        # From non-existent state
        available = node._get_available_transitions('unknown', transitions)
        assert len(available) == 0


class TestPromptChainNode:
    """Tests for prompt chain."""

    def test_prompt_chain_node_exists(self):
        """Verify PromptChainNode is properly defined."""
        from agent_nodes.routing.workflow import PromptChainNode

        assert hasattr(PromptChainNode, 'title')
        assert PromptChainNode.title == 'Prompt Chain'

    def test_substitute_variables(self):
        """Test variable substitution in prompts."""
        from agent_nodes.routing.workflow import PromptChainNode

        node = PromptChainNode.__new__(PromptChainNode)

        result = node._substitute_variables(
            "Hello {name}, you are a {role}.",
            {"name": "Alice", "role": "developer"}
        )

        assert result == "Hello Alice, you are a developer."

    def test_substitute_variables_missing(self):
        """Test variable substitution with missing vars."""
        from agent_nodes.routing.workflow import PromptChainNode

        node = PromptChainNode.__new__(PromptChainNode)

        result = node._substitute_variables(
            "Hello {name}, value is {missing}.",
            {"name": "Bob"}
        )

        assert "Bob" in result
        assert "{missing}" in result  # Not substituted


class TestWorkflowStep:
    """Tests for WorkflowStep dataclass."""

    def test_workflow_step_creation(self):
        """Test creating a workflow step."""
        from agent_nodes.routing.workflow import WorkflowStep

        step = WorkflowStep(
            name="process_input",
            node_type="LLMChat",
            inputs={"prompt": "Hello"},
            outputs=["response"],
            on_success="next_step"
        )

        assert step.name == "process_input"
        assert step.node_type == "LLMChat"
        assert step.on_success == "next_step"


class TestWorkflowDefinition:
    """Tests for WorkflowDefinition dataclass."""

    def test_workflow_definition_creation(self):
        """Test creating a workflow definition."""
        from agent_nodes.routing.workflow import WorkflowDefinition, WorkflowStep

        step = WorkflowStep(
            name="step1",
            node_type="Chat"
        )

        workflow = WorkflowDefinition(
            name="my_workflow",
            description="A test workflow",
            steps=[step],
            initial_step="step1"
        )

        assert workflow.name == "my_workflow"
        assert len(workflow.steps) == 1
        assert workflow.initial_step == "step1"


class TestChainType:
    """Tests for ChainType enum."""

    def test_chain_types_exist(self):
        """Test all chain types exist."""
        from agent_nodes.routing.workflow import ChainType

        assert ChainType.SEQUENTIAL.value == "sequential"
        assert ChainType.BRANCHING.value == "branching"
        assert ChainType.PARALLEL.value == "parallel"
        assert ChainType.RECURSIVE.value == "recursive"
        assert ChainType.TRANSFORM.value == "transform"
