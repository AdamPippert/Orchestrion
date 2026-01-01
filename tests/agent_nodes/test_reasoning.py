"""
Tests for agent reasoning patterns.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../ryven-editor/ryven/example_nodes'))


class TestReActNode:
    """Tests for ReAct reasoning pattern."""

    def test_react_node_exists(self):
        """Verify ReActNode is properly defined."""
        from agent_nodes.agents.reasoning import ReActNode

        assert hasattr(ReActNode, 'title')
        assert ReActNode.title == 'ReAct'

    def test_react_prompt_generation(self):
        """Test ReAct prompt generation."""
        from agent_nodes.agents.reasoning import ReActNode

        node = ReActNode.__new__(ReActNode)
        node.trajectory = []
        node.actions_taken = []
        node.iteration = 0

        tools = [
            {'name': 'search', 'description': 'Search the web'},
            {'name': 'calculate', 'description': 'Perform calculations'},
        ]

        prompt = node._generate_react_prompt(
            task="Find the population of France",
            tools=tools,
            trajectory=[]
        )

        assert "ReAct" in prompt
        assert "Thought" in prompt
        assert "Action" in prompt
        assert "Observation" in prompt
        assert "search" in prompt
        assert "calculate" in prompt

    def test_react_prompt_with_trajectory(self):
        """Test ReAct prompt with existing trajectory."""
        from agent_nodes.agents.reasoning import ReActNode, ReasoningStep, ReasoningStepType

        node = ReActNode.__new__(ReActNode)
        node.trajectory = []
        node.actions_taken = []
        node.iteration = 0

        trajectory = [
            ReasoningStep(
                step_type=ReasoningStepType.THOUGHT,
                content="I need to search for information"
            ),
            ReasoningStep(
                step_type=ReasoningStepType.ACTION,
                content="search('France population')"
            ),
        ]

        prompt = node._generate_react_prompt(
            task="Find the population of France",
            tools=[],
            trajectory=trajectory
        )

        assert "Trajectory so far" in prompt
        assert "Thought" in prompt
        assert "Action" in prompt


class TestPlanExecuteNode:
    """Tests for Plan-and-Execute pattern."""

    def test_plan_execute_node_exists(self):
        """Verify PlanExecuteNode is properly defined."""
        from agent_nodes.agents.reasoning import PlanExecuteNode

        assert hasattr(PlanExecuteNode, 'title')
        assert PlanExecuteNode.title == 'Plan & Execute'

    def test_planning_prompt_generation(self):
        """Test planning prompt generation."""
        from agent_nodes.agents.reasoning import PlanExecuteNode

        node = PlanExecuteNode.__new__(PlanExecuteNode)
        node.plan = None
        node.completed_steps = []

        prompt = node._generate_planning_prompt(
            task="Build a REST API",
            custom_prompt=None
        )

        assert "step-by-step" in prompt.lower()
        assert "Build a REST API" in prompt
        assert "actionable" in prompt.lower()

    def test_custom_planning_prompt(self):
        """Test custom planning prompt."""
        from agent_nodes.agents.reasoning import PlanExecuteNode

        node = PlanExecuteNode.__new__(PlanExecuteNode)
        node.plan = None
        node.completed_steps = []

        custom = "Create a plan for: {task}"
        prompt = node._generate_planning_prompt(
            task="Build a website",
            custom_prompt=custom
        )

        assert "Build a website" in prompt


class TestChainOfThoughtNode:
    """Tests for Chain of Thought pattern."""

    def test_cot_node_exists(self):
        """Verify ChainOfThoughtNode is properly defined."""
        from agent_nodes.agents.reasoning import ChainOfThoughtNode

        assert hasattr(ChainOfThoughtNode, 'title')
        assert ChainOfThoughtNode.title == 'Chain of Thought'

    def test_cot_prompt_numbered(self):
        """Test numbered format CoT prompt."""
        from agent_nodes.agents.reasoning import ChainOfThoughtNode

        node = ChainOfThoughtNode.__new__(ChainOfThoughtNode)

        prompt = node._generate_cot_prompt(
            problem="What is 2 + 2?",
            examples=[],
            require_steps=3,
            format="numbered"
        )

        assert "step by step" in prompt.lower()
        assert "3" in prompt
        assert "Step 1" in prompt
        assert "Final Answer" in prompt

    def test_cot_prompt_structured(self):
        """Test structured format CoT prompt."""
        from agent_nodes.agents.reasoning import ChainOfThoughtNode

        node = ChainOfThoughtNode.__new__(ChainOfThoughtNode)

        prompt = node._generate_cot_prompt(
            problem="Solve this problem",
            examples=[],
            require_steps=2,
            format="structured"
        )

        assert "REASONING" in prompt
        assert "ANSWER" in prompt
        assert "CONFIDENCE" in prompt

    def test_cot_prompt_with_examples(self):
        """Test CoT prompt with few-shot examples."""
        from agent_nodes.agents.reasoning import ChainOfThoughtNode

        node = ChainOfThoughtNode.__new__(ChainOfThoughtNode)

        examples = [
            {
                'problem': 'What is 1+1?',
                'reasoning': 'Adding 1 and 1 gives 2',
                'answer': '2'
            }
        ]

        prompt = node._generate_cot_prompt(
            problem="What is 3+3?",
            examples=examples,
            require_steps=2,
            format="numbered"
        )

        assert "Example 1" in prompt
        assert "1+1" in prompt
        assert "Adding" in prompt


class TestReflectionNode:
    """Tests for Reflection pattern."""

    def test_reflection_node_exists(self):
        """Verify ReflectionNode is properly defined."""
        from agent_nodes.agents.reasoning import ReflectionNode

        assert hasattr(ReflectionNode, 'title')
        assert ReflectionNode.title == 'Reflection'

    def test_reflection_prompt_generation(self):
        """Test reflection prompt generation."""
        from agent_nodes.agents.reasoning import ReflectionNode

        node = ReflectionNode.__new__(ReflectionNode)

        prompt = node._generate_reflection_prompt(
            output="The answer is 42",
            original_task="Find the meaning of life",
            custom_prompt=None,
            criteria=["accuracy", "completeness"]
        )

        assert "42" in prompt
        assert "meaning of life" in prompt
        assert "accuracy" in prompt
        assert "completeness" in prompt
        assert "ISSUES" in prompt
        assert "IMPROVEMENTS" in prompt


class TestTreeOfThoughtsNode:
    """Tests for Tree of Thoughts pattern."""

    def test_tot_node_exists(self):
        """Verify TreeOfThoughtsNode is properly defined."""
        from agent_nodes.agents.reasoning import TreeOfThoughtsNode

        assert hasattr(TreeOfThoughtsNode, 'title')
        assert TreeOfThoughtsNode.title == 'Tree of Thoughts'

    def test_branching_prompt_generation(self):
        """Test branching prompt generation."""
        from agent_nodes.agents.reasoning import TreeOfThoughtsNode

        node = TreeOfThoughtsNode.__new__(TreeOfThoughtsNode)

        prompt = node._generate_branching_prompt(
            problem="Solve a puzzle",
            current_path=[],
            branching_factor=3
        )

        assert "puzzle" in prompt.lower()
        assert "3" in prompt
        assert "Option" in prompt

    def test_branching_prompt_with_path(self):
        """Test branching with existing path."""
        from agent_nodes.agents.reasoning import TreeOfThoughtsNode

        node = TreeOfThoughtsNode.__new__(TreeOfThoughtsNode)

        prompt = node._generate_branching_prompt(
            problem="Complete the maze",
            current_path=["Go left", "Jump over obstacle"],
            branching_factor=2
        )

        assert "Current reasoning path" in prompt
        assert "Go left" in prompt
        assert "Jump over obstacle" in prompt

    def test_evaluation_prompt_generation(self):
        """Test path evaluation prompt generation."""
        from agent_nodes.agents.reasoning import TreeOfThoughtsNode

        node = TreeOfThoughtsNode.__new__(TreeOfThoughtsNode)

        paths = [
            ["Step A1", "Step A2"],
            ["Step B1", "Step B2"],
        ]

        prompt = node._generate_evaluation_prompt(
            problem="Find best route",
            paths=paths,
            custom_prompt=None
        )

        assert "Path 1" in prompt
        assert "Path 2" in prompt
        assert "Rank" in prompt


class TestSelfAskNode:
    """Tests for Self-Ask pattern."""

    def test_self_ask_node_exists(self):
        """Verify SelfAskNode is properly defined."""
        from agent_nodes.agents.reasoning import SelfAskNode

        assert hasattr(SelfAskNode, 'title')
        assert SelfAskNode.title == 'Self-Ask'

    def test_self_ask_prompt_generation(self):
        """Test Self-Ask prompt generation."""
        from agent_nodes.agents.reasoning import SelfAskNode

        node = SelfAskNode.__new__(SelfAskNode)

        prompt = node._generate_self_ask_prompt(
            question="Who wrote the book that inspired the movie Blade Runner?"
        )

        assert "follow-up questions" in prompt.lower()
        assert "Blade Runner" in prompt
        assert "Intermediate answer" in prompt
        assert "final answer" in prompt.lower()


class TestReasoningStep:
    """Tests for ReasoningStep dataclass."""

    def test_reasoning_step_creation(self):
        """Test creating a reasoning step."""
        from agent_nodes.agents.reasoning import ReasoningStep, ReasoningStepType

        step = ReasoningStep(
            step_type=ReasoningStepType.THOUGHT,
            content="I should search for more information"
        )

        assert step.step_type == ReasoningStepType.THOUGHT
        assert step.content == "I should search for more information"
        assert step.timestamp > 0

    def test_all_step_types(self):
        """Test all reasoning step types exist."""
        from agent_nodes.agents.reasoning import ReasoningStepType

        assert ReasoningStepType.THOUGHT.value == "thought"
        assert ReasoningStepType.ACTION.value == "action"
        assert ReasoningStepType.OBSERVATION.value == "observation"
        assert ReasoningStepType.REFLECTION.value == "reflection"
        assert ReasoningStepType.PLAN.value == "plan"


class TestPlan:
    """Tests for Plan dataclass."""

    def test_plan_creation(self):
        """Test creating a plan."""
        from agent_nodes.agents.reasoning import Plan

        plan = Plan(
            goal="Complete the project",
            steps=[
                {"step": 1, "description": "Research"},
                {"step": 2, "description": "Implement"},
            ]
        )

        assert plan.goal == "Complete the project"
        assert len(plan.steps) == 2
        assert plan.current_step == 0
        assert plan.status == "pending"
